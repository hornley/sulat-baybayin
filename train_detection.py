import argparse
import torch
import torchvision
from src.detection.dataset import BBoxDataset
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.shared.train_args import add_common_training_args, dataloader_kwargs_from_args


def get_model(num_classes, weights=None):
    # load a pre-trained model for classification and return
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data_sentences', help='Image root')
    parser.add_argument('--ann', required=True, help='Annotation CSV or COCO JSON')
    add_common_training_args(parser)
    # detection-specific defaults: keep a smaller default batch size
    parser.set_defaults(batch=4, num_workers=4, prefetch_factor=2, out='checkpoints/detection')
    # optimizer / training options
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr-backbone', type=float, default=None, help='Optional LR for backbone params')
    parser.add_argument('--lr-head', type=float, default=None, help='Optional LR for head params')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone parameters')
    parser.add_argument('--schedule', choices=['step', 'cosine', 'none'], default='step')
    parser.add_argument('--lr-step', type=int, default=3)
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    # resume path provided by shared args
    # early-stop is provided by shared args
    parser.add_argument('--val-ann', default=None, help='Optional separate annotation file for validation')
    # dataloader options (provided by shared args)
    # evaluation / performance
    parser.add_argument('--no-batch-eval', action='store_true', help='Disable per-batch quick eval during training')
    args = parser.parse_args()

    args = parser.parse_args()

    ds = BBoxDataset(args.data, ann_file=args.ann)
    classes = ds.classes
    print('Found classes:', classes)
    # build DataLoader kwargs from shared args
    dl_kw = dataloader_kwargs_from_args(args)
    # For detection we need shuffle=True and collate_fn
    loader = DataLoader(ds, shuffle=True, collate_fn=collate_fn, **dl_kw)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=len(classes)+1)
    model.to(device)

    # optionally freeze backbone
    if args.freeze_backbone:
        try:
            for p in model.backbone.parameters():
                p.requires_grad = False
            print('Backbone frozen: only head/RPN params will be trained')
        except Exception:
            print('Warning: could not freeze backbone (unexpected model structure)')

    # build optimizer param groups (optionally use separate LRs for backbone and head)
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

    # scheduler
    if args.schedule == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.schedule == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    else:
        lr_scheduler = None

    # optionally resume
    start_epoch = 0
    if args.resume:
        import os
        if os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
                print('Loaded model state from', args.resume)
            if args.resume_optimizer and 'optimizer_state' in ckpt and 'optimizer' in locals():
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state'])
                    # move optimizer tensors to correct device
                    for state in optimizer.state.values():
                        for k, v in list(state.items()):
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                    print('Loaded optimizer state from', args.resume)
                except Exception:
                    print('Could not load optimizer state (optimizer/model mismatch)')
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch'] + 1
        else:
            print('Resume checkpoint not found:', args.resume)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if (args.amp and device.type == 'cuda') else None

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        import time
        t0 = time.time()
        model.train()
        # accumulators
        total_loss = 0.0
        cls_loss = 0.0
        box_loss = 0.0
        n_batches = 0
        matched = 0
        total_gt = 0
        for imgs, targets in loader:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss_dict = model(imgs, targets)
                    losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            # accumulate loss terms
            batch_total = float(losses.item())
            total_loss += batch_total
            # some keys may not exist depending on model config
            cls = float(loss_dict.get('loss_classifier', torch.tensor(0.0)).item()) if 'loss_classifier' in loss_dict else 0.0
            box = float(loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()) if 'loss_box_reg' in loss_dict else 0.0
            cls_loss += cls
            box_loss += box
            n_batches += 1

            # approximate training "accuracy" (recall at IoU>0.5) using current model predictions
            # run a quick prediction pass (no grad) and match preds to targets
            try:
                model.eval()
                with torch.no_grad():
                    preds = model(imgs)
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
                    # simple greedy matching: for each gt, check if any pred has IoU>0.5 and same label
                    if boxes.shape[0] == 0:
                        continue
                    # compute IoUs
                    def iou_matrix(a, b):
                        # a: Nx4, b: Mx4
                        A = a.shape[0]
                        B = b.shape[0]
                        if A == 0 or B == 0:
                            return torch.zeros((A,B), device=a.device)
                        lt = torch.max(a[:, None, :2], b[None, :, :2])  # [N,M,2]
                        rb = torch.min(a[:, None, 2:], b[None, :, 2:])  # [N,M,2]
                        wh = (rb - lt).clamp(min=0)
                        inter = wh[:,:,0] * wh[:,:,1]
                        area_a = (a[:,2]-a[:,0]).clamp(min=0) * (a[:,3]-a[:,1]).clamp(min=0)
                        area_b = (b[:,2]-b[:,0]).clamp(min=0) * (b[:,3]-b[:,1]).clamp(min=0)
                        union = area_a[:,None] + area_b[None,:] - inter
                        return inter / union.clamp(min=1e-6)

                    ious = iou_matrix(gt_boxes, boxes)
                    for gi in range(gt_boxes.shape[0]):
                        # find preds with same label
                        same_label = (labels == gt_labels[gi])
                        if same_label.sum() == 0:
                            continue
                        ious_g = ious[gi]
                        # consider only same-label preds
                        ious_g = ious_g.clone()
                        ious_g[~same_label] = 0.0
                        if (ious_g > 0.5).any():
                            matched += 1
            finally:
                model.train()

        # compute training averages (needed for logging and as fallback val_loss)
        elapsed = time.time() - t0
        avg_total = total_loss / n_batches if n_batches else 0.0
        avg_cls = cls_loss / n_batches if n_batches else 0.0
        avg_box = box_loss / n_batches if n_batches else 0.0
        acc = matched / total_gt if total_gt else 0.0

        # scheduler step
        if lr_scheduler is not None:
            try:
                lr_scheduler.step()
            except Exception:
                pass

        # optional validation pass (if separate val annotations provided)
        val_loss = None
        if args.val_ann:
            val_ds = BBoxDataset(args.data, ann_file=args.val_ann)
            val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn,
                                    num_workers=args.num_workers, pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor)
            model.eval()
            total_val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = list(img.to(device) for img in imgs)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(imgs, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    total_val_loss += float(losses.item())
                    n_val_batches += 1
            if n_val_batches:
                val_loss = total_val_loss / n_val_batches
                print(f'Validation: val_loss={val_loss:.4f}')
            model.train()
        else:
            # fallback: use training loss as a proxy for validation when no val set is provided
            val_loss = avg_total
        print(f'Epoch {epoch+1}/{args.epochs}: train_loss={avg_total:.4f} cls_loss={avg_cls:.4f} box_loss={avg_box:.4f} acc={acc:.4f} time={elapsed:.1f}s')

        # early stopping logic (monitor val_loss)
        improved = False
        if val_loss is not None:
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_no_improve = 0
                improved = True
            else:
                epochs_no_improve += 1

        # save checkpoint when improvement is observed (or first epoch)
        import os
        os.makedirs(args.out, exist_ok=True)
        if improved or epoch == start_epoch:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'classes': classes}, os.path.join(args.out, 'best.pth'))
            print(f'Saved checkpoint to {os.path.join(args.out, "best.pth")}')

        # check early stopping
        if args.early_stop and epochs_no_improve >= args.early_stop:
            print(f'Early stopping: no improvement in val_loss for {args.early_stop} epochs')
            break

from src.detection.train import main


if __name__ == '__main__':
    main()
