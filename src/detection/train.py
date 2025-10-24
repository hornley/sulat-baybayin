import argparse
import os
import time
import torch
import torchvision
import torchvision.transforms as T
import random
import sys
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.detection.dataset import BBoxDataset
from src.shared.train_args import add_common_training_args, dataloader_kwargs_from_args
from src.shared.utils import save_checkpoint
from src.shared.config_manager import generate_yaml_template, load_yaml_config, merge_configs, wait_for_user_edit


def backup_to_gdrive(output_path, gdrive_path):
    """Backup checkpoint directory to Google Drive.
    
    Args:
        output_path: Local checkpoint path (e.g., checkpoints/detection/colab_run11/stage1)
        gdrive_path: Google Drive backup path (e.g., /content/drive/MyDrive/SulatBaybayin/)
    """
    try:
        import shutil
        # Extract the run name from output path
        out_parts = os.path.normpath(output_path).split(os.sep)
        if 'detection' in out_parts:
            det_idx = out_parts.index('detection')
            if det_idx + 1 < len(out_parts):
                run_name = out_parts[det_idx + 1]
                # Find the base checkpoint directory (e.g., checkpoints/detection/colab_run11)
                base_checkpoint_dir = os.path.join(*out_parts[:det_idx + 2])
                
                # Create backup destination
                backup_dest = os.path.join(gdrive_path, run_name)
                
                # Copy the entire run directory
                if os.path.exists(base_checkpoint_dir):
                    os.makedirs(gdrive_path, exist_ok=True)
                    if os.path.exists(backup_dest):
                        shutil.rmtree(backup_dest)
                    shutil.copytree(base_checkpoint_dir, backup_dest)
                    return True, backup_dest
                else:
                    return False, f"Source not found: {base_checkpoint_dir}"
        return False, f"Could not parse path: {output_path}"
    except Exception as e:
        return False, str(e)


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
    
    # === YAML CONFIG OPTIONS ===
    parser.add_argument('--args-input', default=None, help='Path to YAML config file. If not exists, will generate template and pause for user to edit')
    parser.add_argument('--no-wait', action='store_true', help='Do not pause for user to edit generated YAML template (use defaults)')
    parser.add_argument('--regen-args', action='store_true', help='Force regeneration of YAML template even if file exists')
    
    parser.add_argument('--data', required=True, help='Image root folder')
    parser.add_argument('--ann', required=True, help='CSV or COCO annotations file')
    add_common_training_args(parser)
    parser.add_argument('--save-last', action='store_true', help='Save last epoch checkpoint to last.pth (won\'t overwrite best.pth)')
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
    # real/synthetic mixing
    parser.add_argument('--real-data', default=None, help='Root folder for photographed/real images to mix with synthetic')
    parser.add_argument('--real-ann', default=None, help='Annotations CSV/COCO for real images')
    parser.add_argument('--real-weight', type=float, default=0.2, help='Fraction of samples from real data per epoch (0-1)')
    parser.add_argument('--mix-strategy', choices=['concat', 'alternating', 'quotas'], default='quotas', help='How to mix synthetic and real datasets')
    # early stopping options
    parser.add_argument('--early-stop-patience', type=int, default=0, help='Number of epochs with no improvement after which training will be stopped (0 disables)')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0, help='Minimum change in monitored metric to qualify as improvement')
    parser.add_argument('--early-stop-monitor', choices=['val_loss', 'train_loss', 'acc'], default='val_loss', help='Metric to monitor for early stopping')
    # google drive backup option
    parser.add_argument('--gdrive-backup', default=None, help='Google Drive path to backup checkpoints (e.g., /content/drive/MyDrive/SulatBaybayin/)')
    args = parser.parse_args()

    # === YAML CONFIG LOADING ===
    if args.args_input is not None:
        yaml_path = args.args_input
        
        # Check if user wants to force regenerate the template
        if args.regen_args and os.path.exists(yaml_path):
            print(f'Regenerating YAML template at {yaml_path} due to --regen-args flag')
            os.remove(yaml_path)
        
        # If YAML file doesn't exist, generate template and optionally wait for user to edit
        if not os.path.exists(yaml_path):
            print(f'YAML config file not found at {yaml_path}')
            print('Generating template with current defaults...')
            
            # Extract all args except the YAML-specific ones
            yaml_args = {k: v for k, v in vars(args).items() 
                        if k not in ('args_input', 'no_wait', 'regen_args')}
            
            # Generate template
            generate_yaml_template(yaml_path, yaml_args)
            print(f'✓ Generated template: {yaml_path}')
            
            # Wait for user to edit unless --no-wait is specified
            if not args.no_wait:
                print()
                print('Please edit the YAML file to configure your parameters.')
                print('Press Enter when ready to continue...')
                wait_for_user_edit(yaml_path)
            else:
                print('Continuing with default values (--no-wait specified)')
        
        # Load YAML config and merge with CLI args (CLI takes precedence)
        print(f'Loading YAML config from {yaml_path}...')
        yaml_config = load_yaml_config(yaml_path)
        
        # Merge: YAML provides base values, CLI overrides
        merged = merge_configs(yaml_config, vars(args))
        
        # Update args namespace with merged values
        for key, value in merged.items():
            setattr(args, key, value)
        
        print('✓ YAML config loaded and merged with CLI arguments')

    # If user asked to monitor val_loss but didn't provide a validation annotation file, warn them
    if args.early_stop_monitor == 'val_loss' and not args.val_ann:
        print('Warning: --early-stop-monitor val_loss selected but no --val-ann provided; val_loss will be None and early-stopping on val_loss will be disabled.', file=sys.stderr)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl_kwargs = dataloader_kwargs_from_args(args)

    # build dataset and dataloader (root, ann_file=...)
    # ensure dataset yields torch tensors (ToTensor) so .to(device) works
    default_transforms = T.ToTensor()
    train_ds = BBoxDataset(args.data, ann_file=args.ann, transforms=default_transforms)
    # dataloader_kwargs_from_args returns a dict with batch_size and other kwargs
    dl_kwargs_local = dict(dl_kwargs)
    batch_size = dl_kwargs_local.pop('batch_size', args.batch)

    # If real data is provided, create a combined dataset and use a WeightedRandomSampler to mix
    if args.real_data and args.real_ann:
        real_ds = BBoxDataset(args.real_data, ann_file=args.real_ann, transforms=default_transforms)
        # concat datasets via simple index mapping: synthetic indices [0..N1-1], real indices [N1..N1+N2-1]
        from torch.utils.data import Dataset

        class ConcatIndexDataset(Dataset):
            def __init__(self, ds_list):
                self.ds_list = ds_list
                self.lengths = [len(d) for d in ds_list]
                self.cum_lengths = [0]
                for l in self.lengths:
                    self.cum_lengths.append(self.cum_lengths[-1] + l)

            def __len__(self):
                return sum(self.lengths)

            def __getitem__(self, idx):
                # find dataset
                for di in range(len(self.ds_list)):
                    if idx < self.cum_lengths[di+1]:
                        local_idx = idx - self.cum_lengths[di]
                        return self.ds_list[di][local_idx]
                raise IndexError

        combined = ConcatIndexDataset([train_ds, real_ds])
        n_synth = len(train_ds)
        n_real = len(real_ds)
        assert n_synth > 0 and n_real > 0, 'Both synthetic and real datasets must be non-empty'

        # mixing strategies
        if args.mix_strategy == 'concat':
            # simple concatenation: synthetic then real; shuffle controlled by dataloader shuffle
            train_loader = torch.utils.data.DataLoader(combined, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, **dl_kwargs_local)
            mix_log_path = os.path.join(args.out, 'mix_log_concat.csv')
        elif args.mix_strategy == 'alternating':
            # alternating batches: construct an index list where batches alternate between datasets
            # Build indices for each dataset and then interleave batch blocks
            synth_indices = list(range(0, n_synth))
            real_indices = list(range(n_synth, n_synth + n_real))
            # shuffle inside each set
            random.shuffle(synth_indices)
            random.shuffle(real_indices)
            # build an interleaved index list of length = len(combined)
            idxs = []
            si = 0; ri = 0
            toggle = True
            while si < len(synth_indices) or ri < len(real_indices):
                if toggle and si < len(synth_indices):
                    idxs.append(synth_indices[si]); si += 1
                elif not toggle and ri < len(real_indices):
                    idxs.append(real_indices[ri]); ri += 1
                else:
                    # drain remaining
                    if si < len(synth_indices):
                        idxs.append(synth_indices[si]); si += 1
                    if ri < len(real_indices):
                        idxs.append(real_indices[ri]); ri += 1
                toggle = not toggle
            from torch.utils.data import Subset
            train_loader = torch.utils.data.DataLoader(combined, batch_size=batch_size, sampler=torch.utils.data.sampler.SequentialSampler(idxs), collate_fn=collate_fn, **dl_kwargs_local)
            mix_log_path = os.path.join(args.out, 'mix_log_alternating.csv')
        else:
            # quotas: deterministic per-epoch quotas without replacement
            total = len(combined)
            # calculate quotas per epoch
            real_quota = int(round(args.real_weight * total))
            synth_quota = total - real_quota
            # sample without replacement deterministically each epoch: we'll create a list of indices for one epoch
            synth_indices = list(range(0, n_synth))
            real_indices = list(range(n_synth, n_synth + n_real))
            random.shuffle(synth_indices)
            random.shuffle(real_indices)
            # take quotas, if we run out, wrap around (but within epoch keep counts exact)
            chosen = []
            si = 0
            ri = 0
            for _ in range(synth_quota):
                if si >= len(synth_indices):
                    si = 0
                    random.shuffle(synth_indices)
                chosen.append(synth_indices[si]); si += 1
            for _ in range(real_quota):
                if ri >= len(real_indices):
                    ri = 0
                    random.shuffle(real_indices)
                chosen.append(real_indices[ri]); ri += 1
            # shuffle final chosen order to mix within epoch but quotas satisfied
            random.shuffle(chosen)
            from torch.utils.data import Subset
            epoch_subset = Subset(combined, chosen)
            train_loader = torch.utils.data.DataLoader(epoch_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, **dl_kwargs_local)
            mix_log_path = os.path.join(args.out, 'mix_log_quotas.csv')
    else:
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

    # prepare mix logging
    mix_log = []
    for epoch in range(start_epoch, args.epochs):
        avg_total, avg_cls, avg_box, matched, total_gt, elapsed = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler, skip_batch_eval=args.no_batch_eval)
        # if using quotas strategy we recorded a subset; compute how many samples from each source were used
        if args.real_data and args.real_ann and args.mix_strategy in ('alternating', 'quotas'):
            # count how many samples in the current train_loader came from real vs synth by inspecting dataset indices if available
            try:
                # for SequentialSampler or Subset, inspect underlying indices
                indices_used = []
                if hasattr(train_loader.dataset, 'indices'):
                    indices_used = list(train_loader.dataset.indices)
                elif hasattr(train_loader.batch_sampler, 'sampler') and hasattr(train_loader.batch_sampler.sampler, 'data_source'):
                    # fallback; may not be necessary
                    indices_used = list(range(len(train_loader.dataset)))
                else:
                    indices_used = []
                n_real_used = sum(1 for idx in indices_used if idx >= n_synth)
                n_synth_used = sum(1 for idx in indices_used if idx < n_synth)
                mix_log.append({'epoch': epoch, 'n_synth': n_synth_used, 'n_real': n_real_used})
            except Exception:
                pass
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
                # save best-seen checkpoint (keeps best observed by the monitored metric during training)
                save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'classes': classes}, os.path.join(args.out, f'best_seen.pth'))
                print(f'New best {args.early_stop_monitor}={best_monitored:.4f} at epoch {epoch+1}, saved best_seen.pth')
            else:
                epochs_no_improve += 1
                print(f'No improvement in {args.early_stop_monitor} for {epochs_no_improve} epochs (best: {best_monitored})')

            if epochs_no_improve >= args.early_stop_patience and args.early_stop_patience > 0:
                print(f'Early stopping: no improvement in {args.early_stop_monitor} for {epochs_no_improve} epochs (patience={args.early_stop_patience})')
                # save final checkpoint
                save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'classes': classes}, os.path.join(args.out, f'final_epoch_{epoch}.pth'))
                # also save an explicit early-stop checkpoint
                save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'classes': classes}, os.path.join(args.out, 'early_stop.pth'))
                print(f'Saved early-stop checkpoint to early_stop.pth')
                break

        # scheduler step
        if lr_scheduler is not None:
            try:
                lr_scheduler.step()
            except Exception:
                pass

        # Save last.pth each epoch when requested (overwrite)
        if args.save_last:
            save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'classes': classes}, os.path.join(args.out, 'last.pth'))
            print(f'Saved last.pth (epoch {epoch+1})')
            
            # Backup to Google Drive after each epoch if specified
            if args.gdrive_backup:
                success, result = backup_to_gdrive(args.out, args.gdrive_backup)
                if success:
                    print(f'✓ Backed up to GDrive: {result}')
                else:
                    print(f'⚠ GDrive backup failed: {result}')

    # At training end (normal completion or early stop), write final checkpoint to best.pth
    try:
        save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'classes': classes}, os.path.join(args.out, 'best.pth'))
        print('Saved final checkpoint to best.pth')
    except Exception:
        print('Warning: could not write final best.pth')

    # Final backup to Google Drive
    if args.gdrive_backup:
        print(f'\n{"="*70}')
        print('Final backup to Google Drive...')
        print(f'{"="*70}')
        success, result = backup_to_gdrive(args.out, args.gdrive_backup)
        if success:
            print(f'✓ Successfully backed up to: {result}')
        else:
            print(f'⚠ Backup failed: {result}')
        print(f'{"="*70}\n')

    # write mix log if collected
    try:
        if 'mix_log' in locals() and len(mix_log) and 'mix_log_path' in locals():
            import csv as _csv
            os.makedirs(os.path.dirname(mix_log_path), exist_ok=True)
            with open(mix_log_path, 'w', newline='', encoding='utf8') as mf:
                w = _csv.writer(mf)
                w.writerow(['epoch', 'n_synth', 'n_real'])
                for r in mix_log:
                    w.writerow([r.get('epoch'), r.get('n_synth'), r.get('n_real')])
            print('Wrote mix log to', mix_log_path)
    except Exception:
        pass


if __name__ == '__main__':
    main()
