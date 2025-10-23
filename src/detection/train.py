import argparse
import os
import time
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.detection.dataset import BBoxDataset
from src.shared.train_args import add_common_training_args, dataloader_kwargs_from_args
from src.shared.utils import save_checkpoint


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, scaler=None, skip_batch_eval=False):
    model.train()
    # accumulators
    total_loss = 0.0
    cls_loss = 0.0
    box_loss = 0.0
    n_batches = 0
    matched = 0
    total_gt = 0
    import time
    t0 = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        if scaler is not None:
            dev = device if isinstance(device, str) else device.type
            with torch.amp.autocast(device_type=dev):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        # accumulate loss terms
        batch_total = float(losses.item())
        total_loss += batch_total
        cls = float(loss_dict.get('loss_classifier', torch.tensor(0.0)).item()) if 'loss_classifier' in loss_dict else 0.0
        box = float(loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()) if 'loss_box_reg' in loss_dict else 0.0
        cls_loss += cls
        box_loss += box
        n_batches += 1

        # approximate training "accuracy" (recall at IoU>0.5) using current model predictions
        if not skip_batch_eval:
            try:
                model.eval()
                with torch.no_grad():
                    preds = model(images)
                # compute matches per image
                for pred, tgt in zip(preds, targets):
                    gt_boxes = tgt.get('boxes', torch.zeros((0,4), device=device))
                    gt_labels = tgt.get('labels', torch.zeros((0,), dtype=torch.int64, device=device))
                    total_gt += gt_boxes.shape[0]
                    if gt_boxes.shape[0] == 0:
                        continue
                    # filter high-confidence predictions
                    scores = pred.get('scores', torch.zeros((0,), device=device))
                    boxes = pred.get('boxes', torch.zeros((0,4), device=device))
                    labels = pred.get('labels', torch.zeros((0,), device=device))
                    keep = scores > 0.5
                    boxes = boxes[keep]
                    labels = labels[keep]
                    if boxes.shape[0] == 0:
                        continue
                    def iou_matrix(a, b):
                        A = a.shape[0]
                        B = b.shape[0]
                        if A == 0 or B == 0:
                            return torch.zeros((A,B), device=a.device)
                        lt = torch.max(a[:, None, :2], b[None, :, :2])
                        rb = torch.min(a[:, None, 2:], b[None, :, 2:])
                        wh = (rb - lt).clamp(min=0)
                        inter = wh[:,:,0] * wh[:,:,1]
                        area_a = (a[:,2]-a[:,0]).clamp(min=0) * (a[:,3]-a[:,1]).clamp(min=0)
                        area_b = (b[:,2]-b[:,0]).clamp(min=0) * (b[:,3]-b[:,1]).clamp(min=0)
                        union = area_a[:,None] + area_b[None,:] - inter
                        return inter / union.clamp(min=1e-6)
                    ious = iou_matrix(gt_boxes, boxes)
                    for gi in range(gt_boxes.shape[0]):
                        same_label = (labels == gt_labels[gi])
                        if same_label.sum() == 0:
                            continue
                        ious_g = ious[gi].clone()
                        ious_g[~same_label] = 0.0
                        if (ious_g > 0.5).any():
                            matched += 1
            finally:
                model.train()

        # if i % print_freq == 0:
        #     print(f'Epoch[{epoch}] Iter[{i}/{len(data_loader)}] Loss: {losses.item():.4f}')

    elapsed = time.time() - t0
    avg_total = total_loss / n_batches if n_batches else 0.0
    avg_cls = cls_loss / n_batches if n_batches else 0.0
    avg_box = box_loss / n_batches if n_batches else 0.0
    return avg_total, avg_cls, avg_box, matched, total_gt, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Image root folder')
    parser.add_argument('--ann', required=True, help='CSV or COCO annotations file')
    add_common_training_args(parser)
    # detection-specific optimizer / training options
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr-backbone', type=float, default=None, help='Optional LR for backbone params')
    parser.add_argument('--lr-head', type=float, default=None, help='Optional LR for head params')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone parameters')
    parser.add_argument('--schedule', choices=['step', 'cosine', 'none'], default='step')
    parser.add_argument('--lr-step', type=int, default=3)
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--val-ann', default=None, help='Optional separate annotation file for validation')
    parser.add_argument('--no-batch-eval', action='store_true', help='Disable per-batch quick eval during training')
    # early stopping options
    parser.add_argument('--early-stop-patience', type=int, default=0, help='Number of epochs with no improvement after which training will be stopped (0 disables)')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0, help='Minimum change in monitored metric to qualify as improvement')
    parser.add_argument('--early-stop-monitor', choices=['val_loss', 'train_loss', 'acc'], default='val_loss', help='Metric to monitor for early stopping')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl_kwargs = dataloader_kwargs_from_args(args)

    # build dataset and dataloader (root, ann_file=...)
    # ensure dataset yields torch tensors (ToTensor) so .to(device) works
    default_transforms = T.ToTensor()
    train_ds = BBoxDataset(args.data, ann_file=args.ann, transforms=default_transforms)
    # dataloader_kwargs_from_args returns a dict with batch_size and other kwargs
    dl_kwargs_local = dict(dl_kwargs)
    batch_size = dl_kwargs_local.pop('batch_size', args.batch)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, **dl_kwargs_local)

    classes = train_ds.classes
    model = get_model(len(classes) + 1)
    model.to(device)

    # optionally freeze backbone
    if args.freeze_backbone:
        try:
            for p in model.backbone.parameters():
                p.requires_grad = False
            print('Backbone frozen: only head/RPN params will be trained')
        except Exception:
            print('Warning: could not freeze backbone (unexpected model structure)')

    # build optimizer param groups (optional separate LR for backbone/head)
    backbone_ids = {id(p) for p in model.backbone.parameters()} if hasattr(model, 'backbone') else set()
    backbone_params = []
    head_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in backbone_ids:
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if backbone_params and args.lr_backbone is not None:
        param_groups.append({'params': backbone_params, 'lr': args.lr_backbone})
    elif backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.lr})
    if head_params and args.lr_head is not None:
        param_groups.append({'params': head_params, 'lr': args.lr_head})
    elif head_params:
        param_groups.append({'params': head_params, 'lr': args.lr})

    if param_groups:
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler() if (args.amp and device == 'cuda') else None

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        if 'model_state' in ck:
            model.load_state_dict(ck['model_state'])
            print('Loaded model state from', args.resume)
        if args.resume_optimizer and 'optimizer_state' in ck:
            try:
                optimizer.load_state_dict(ck['optimizer_state'])
                for state in optimizer.state.values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                print('Loaded optimizer state from', args.resume)
            except Exception:
                print('Could not load optimizer state (optimizer/model mismatch)')
        if 'epoch' in ck:
            start_epoch = ck['epoch'] + 1

    os.makedirs(args.out, exist_ok=True)

    # scheduler
    if args.schedule == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.schedule == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    else:
        lr_scheduler = None

    for epoch in range(start_epoch, args.epochs):
        avg_total, avg_cls, avg_box, matched, total_gt, elapsed = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler, skip_batch_eval=args.no_batch_eval)
        acc = matched / total_gt if total_gt else 0.0
        # print baseline epoch training summary; validation printed below if present
        print(f'Epoch {epoch+1}/{args.epochs}: train_loss={avg_total:.4f} cls_loss={avg_cls:.4f} box_loss={avg_box:.4f} acc={acc:.4f} time={elapsed:.1f}s')

        # optional validation when val_ann provided
        val_loss = None
        if args.val_ann:
            val_ds = BBoxDataset(args.data, ann_file=args.val_ann, transforms=default_transforms)
            v_dl_kwargs = dict(batch_size=batch_size, shuffle=False, collate_fn=collate_fn, **{k: v for k, v in dl_kwargs_local.items() if k != 'shuffle'})
            val_loader = torch.utils.data.DataLoader(val_ds, **v_dl_kwargs)
            model.eval()
            total_val = 0.0
            n_val = 0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = list(img.to(device) for img in imgs)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(imgs, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    total_val += float(losses.item())
                    n_val += 1
            if n_val:
                val_loss = total_val / n_val
                print(f'Validation: val_loss={val_loss:.4f}')
        # if validation was run, include val_loss in a combined summary line
        if val_loss is not None:
            print(f'Epoch {epoch+1}/{args.epochs}: train_loss={avg_total:.4f} cls_loss={avg_cls:.4f} box_loss={avg_box:.4f} acc={acc:.4f} val_loss={val_loss:.4f} time={elapsed:.1f}s')
            model.train()

        # determine monitored metric for early stopping
        monitored = None
        if args.early_stop_monitor == 'val_loss':
            monitored = val_loss if val_loss is not None else None
        elif args.early_stop_monitor == 'train_loss':
            monitored = avg_total
        elif args.early_stop_monitor == 'acc':
            monitored = acc

        # early stopping bookkeeping
        if epoch == start_epoch:
            best_monitored = None
            best_epoch = epoch
            epochs_no_improve = 0
        # initialize best_monitored on first available value
        if 'best_monitored' not in locals() or best_monitored is None:
            if monitored is not None:
                best_monitored = monitored
                best_epoch = epoch
                epochs_no_improve = 0

        # check improvement
        if args.early_stop_patience and monitored is not None:
            improved = False
            if args.early_stop_monitor == 'acc':
                if monitored > (best_monitored + args.early_stop_min_delta):
                    improved = True
            else:
                # lower is better for losses
                if monitored < (best_monitored - args.early_stop_min_delta):
                    improved = True

            if improved:
                best_monitored = monitored
                best_epoch = epoch
                epochs_no_improve = 0
                # save best checkpoint
                save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'classes': classes}, os.path.join(args.out, f'best.pth'))
                print(f'New best {args.early_stop_monitor}={best_monitored:.4f} at epoch {epoch+1}, saved best.pth')
            else:
                epochs_no_improve += 1
                print(f'No improvement in {args.early_stop_monitor} for {epochs_no_improve} epochs (best: {best_monitored})')

            if epochs_no_improve >= args.early_stop_patience and args.early_stop_patience > 0:
                print(f'Early stopping: no improvement in {args.early_stop_monitor} for {epochs_no_improve} epochs (patience={args.early_stop_patience})')
                # save final checkpoint
                save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'classes': classes}, os.path.join(args.out, f'final_epoch_{epoch}.pth'))
                break

        # scheduler step
        if lr_scheduler is not None:
            try:
                lr_scheduler.step()
            except Exception:
                pass

        save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'classes': classes}, os.path.join(args.out, f'best.pth'))


if __name__ == '__main__':
    main()
