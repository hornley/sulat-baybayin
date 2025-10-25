import os
import argparse
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from src.classification.dataset import make_dataloaders
from src.classification.model import BaybayinClassifier
from src.shared.utils import save_checkpoint
from src.shared.train_args import add_common_training_args, dataloader_kwargs_from_args
from src.shared.config_manager import generate_yaml_template, load_yaml_config, merge_configs, wait_for_user_edit


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            dev = device if isinstance(device, str) else device.type
            with torch.amp.autocast(device_type=dev):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    
    # === YAML CONFIG OPTIONS ===
    parser.add_argument('--args-input', default=None, help='Path to YAML config file. If not exists, will generate template and pause for user to edit')
    parser.add_argument('--no-wait', action='store_true', help='Do not pause for user to edit generated YAML template (use defaults)')
    parser.add_argument('--regen-args', action='store_true', help='Force regeneration of YAML template even if file exists')
    
    parser.add_argument('--data', required=True, help='Root folder with class subfolders')
    # common args
    add_common_training_args(parser)
    # classification-specific args
    parser.add_argument('--lr-backbone', type=float, default=None, help='Optional lower LR for backbone when fine-tuning')
    parser.add_argument('--lr-head', type=float, default=None, help='Optional LR for head/classifier when fine-tuning')
    parser.add_argument('--schedule', choices=['none', 'cosine', 'plateau'], default='none', help='LR schedule to use')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (by val_loss); set 0 to disable')
    # backward-compatible alias used previously by some scripts
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--augment', action='store_true', help='Enable standard geometric/color augmentations')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone layers and train only the classifier')
    parser.add_argument('--min-count', type=int, default=5, help='Minimum number of images required for a class to be included')
    parser.add_argument('--weights', type=str, default='default', help="Which pretrained weights to use: 'default', 'v1', or 'none'")
    
    # === PAPER TEXTURE AUGMENTATION ===
    parser.add_argument('--aug-paper-prob', type=float, default=0.0, help='Probability to apply paper texture augmentation (0..1)')
    parser.add_argument('--aug-paper-type-probs', type=str, default='0.6,0.2,0.2', help='Probabilities for [white, yellow-paper, dotted] as comma-separated (should sum to 1.0)')
    parser.add_argument('--aug-paper-texture-probs', type=str, default='0.5,0.3,0.2', help='Probabilities for [plain, grainy, crumpled] as comma-separated (should sum to 1.0)')
    parser.add_argument('--aug-paper-strength-min', type=float, default=0.2, help='Minimum paper blend strength (0..1)')
    parser.add_argument('--aug-paper-strength-max', type=float, default=0.4, help='Maximum paper blend strength (0..1)')
    parser.add_argument('--aug-paper-yellow-strength-min', type=float, default=0.3, help='Minimum yellow-paper strength (0..1)')
    parser.add_argument('--aug-paper-yellow-strength-max', type=float, default=0.5, help='Maximum yellow-paper strength (0..1)')
    parser.add_argument('--aug-crumple-strength-min', type=float, default=1.0, help='Minimum crumple warp strength')
    parser.add_argument('--aug-crumple-strength-max', type=float, default=3.0, help='Maximum crumple warp strength')
    parser.add_argument('--aug-crumple-mesh-overlap', type=int, default=2, help='Pixel overlap for crumple mesh tiles')
    
    # === PAPER LINES AUGMENTATION ===
    parser.add_argument('--aug-lines-prob', type=float, default=0.0, help='Probability to overlay ruled paper lines (0..1)')
    parser.add_argument('--aug-line-spacing-min', type=int, default=24, help='Minimum pixel spacing between lines')
    parser.add_argument('--aug-line-spacing-max', type=int, default=32, help='Maximum pixel spacing between lines')
    parser.add_argument('--aug-line-opacity-min', type=int, default=30, help='Minimum line opacity (0-255)')
    parser.add_argument('--aug-line-opacity-max', type=int, default=60, help='Maximum line opacity (0-255)')
    parser.add_argument('--aug-line-thickness-min', type=int, default=1, help='Minimum line thickness in pixels')
    parser.add_argument('--aug-line-thickness-max', type=int, default=2, help='Maximum line thickness in pixels')
    parser.add_argument('--aug-line-jitter-min', type=int, default=1, help='Minimum vertical jitter per line')
    parser.add_argument('--aug-line-jitter-max', type=int, default=3, help='Maximum vertical jitter per line')
    parser.add_argument('--aug-line-color', type=str, default='0,0,0', help='RGB color for lines as comma-separated ints')
    
    # === LIGHTING AUGMENTATION ===
    parser.add_argument('--aug-lighting-prob', type=float, default=0.0, help='Probability to apply lighting variations (0..1)')
    parser.add_argument('--aug-lighting-modes', type=str, default='normal,bright,dim,shadows', help='Comma-separated lighting modes to sample from')
    parser.add_argument('--aug-brightness-jitter', type=float, default=0.03, help='Brightness jitter amount for normal lighting mode')
    parser.add_argument('--aug-contrast-jitter', type=float, default=0.03, help='Contrast jitter amount for normal lighting mode')
    parser.add_argument('--aug-shadow-intensity-min', type=float, default=0.0, help='Minimum shadow intensity for shadows mode (0..1)')
    parser.add_argument('--aug-shadow-intensity-max', type=float, default=0.3, help='Maximum shadow intensity for shadows mode (0..1)')
    
    # === DOTTED PAPER OPTIONS ===
    parser.add_argument('--aug-dot-size', type=int, default=1, help='Radius for dotted paper dots')
    parser.add_argument('--aug-dot-opacity', type=int, default=50, help='Opacity for dotted paper dots (0-255)')
    parser.add_argument('--aug-dot-spacing', type=int, default=18, help='Spacing in pixels between dots')
    
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

    # device selection
    device = args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

    # build paper augmentation if any paper aug flags are enabled
    paper_aug = None
    if args.aug_paper_prob > 0 or args.aug_lines_prob > 0 or args.aug_lighting_prob > 0:
        from src.classification.dataset import PaperAugmentation
        
        # parse probability lists
        def parse_probs(s):
            return [float(x.strip()) for x in s.split(',')]
        
        # parse color
        def parse_color(s):
            parts = [int(x.strip()) for x in s.split(',')]
            return tuple(parts)
        
        paper_type_probs = parse_probs(args.aug_paper_type_probs)
        paper_texture_probs = parse_probs(args.aug_paper_texture_probs)
        lighting_modes = [m.strip() for m in args.aug_lighting_modes.split(',')]
        line_color = parse_color(args.aug_line_color)
        
        paper_aug = PaperAugmentation(
            # paper texture
            paper_prob=args.aug_paper_prob,
            paper_type_probs=paper_type_probs,
            paper_texture_probs=paper_texture_probs,
            paper_strength_min=args.aug_paper_strength_min,
            paper_strength_max=args.aug_paper_strength_max,
            paper_yellow_strength_min=args.aug_paper_yellow_strength_min,
            paper_yellow_strength_max=args.aug_paper_yellow_strength_max,
            crumple_strength_min=args.aug_crumple_strength_min,
            crumple_strength_max=args.aug_crumple_strength_max,
            crumple_mesh_overlap=args.aug_crumple_mesh_overlap,
            # paper lines
            lines_prob=args.aug_lines_prob,
            line_spacing_min=args.aug_line_spacing_min,
            line_spacing_max=args.aug_line_spacing_max,
            line_opacity_min=args.aug_line_opacity_min,
            line_opacity_max=args.aug_line_opacity_max,
            line_thickness_min=args.aug_line_thickness_min,
            line_thickness_max=args.aug_line_thickness_max,
            line_jitter_min=args.aug_line_jitter_min,
            line_jitter_max=args.aug_line_jitter_max,
            line_color=line_color,
            # lighting
            lighting_prob=args.aug_lighting_prob,
            lighting_modes=lighting_modes,
            brightness_jitter=args.aug_brightness_jitter,
            contrast_jitter=args.aug_contrast_jitter,
            shadow_intensity_min=args.aug_shadow_intensity_min,
            shadow_intensity_max=args.aug_shadow_intensity_max,
            # dotted paper
            dot_size=args.aug_dot_size,
            dot_opacity=args.aug_dot_opacity,
            dot_spacing=args.aug_dot_spacing,
        )
        
        print(f"Paper augmentation enabled:")
        print(f"  - Paper texture prob: {args.aug_paper_prob}")
        print(f"  - Lines prob: {args.aug_lines_prob}")
        print(f"  - Lighting prob: {args.aug_lighting_prob}")

    # build dataloaders with DataLoader kwargs from common args
    dl_kwargs = dataloader_kwargs_from_args(args)
    train_loader, val_loader, classes = make_dataloaders(
        args.data, 
        batch_size=dl_kwargs.get('batch_size', args.batch), 
        img_size=args.img_size, 
        augment=args.augment, 
        min_count=args.min_count,
        paper_aug=paper_aug
    )

    # map weights option to torchvision enum when available
    weights_arg = None
    if args.weights.lower() != 'none':
        try:
            from torchvision.models import ResNet18_Weights
            if args.weights.lower() == 'v1':
                weights_arg = ResNet18_Weights.IMAGENET1K_V1
            else:
                weights_arg = ResNet18_Weights.DEFAULT
        except Exception:
            weights_arg = None

    model = BaybayinClassifier(num_classes=len(classes), weights=weights_arg)
    model = model.to(device)

    if args.freeze_backbone:
        for name, param in model.backbone.named_parameters():
            if name.startswith('fc.'):
                continue
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler() if (args.amp and (device == 'cuda' or (not isinstance(device, str) and device.type == 'cuda'))) else None
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_groups = None
    if args.lr_backbone is not None or args.lr_head is not None:
        backbone_params = []
        head_params = []
        try:
            for name, p in model.backbone.named_parameters():
                if 'fc' in name:
                    if p.requires_grad:
                        head_params.append(p)
                else:
                    if p.requires_grad:
                        backbone_params.append(p)
        except Exception:
            backbone_params = [p for p in model.parameters() if p.requires_grad]
            head_params = []

        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': args.lr_backbone or args.lr})
        if head_params:
            param_groups.append({'params': head_params, 'lr': args.lr_head or args.lr})

    if param_groups:
        optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume (load model and optimizer state if requested)
    start_epoch = 0
    if getattr(args, 'resume', None):
        if os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
                print('Loaded model state from', args.resume)
            if args.resume_optimizer and 'optimizer_state' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state'])
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

    # scheduler
    scheduler = None
    if args.schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    elif args.schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    os.makedirs(args.out, exist_ok=True)

    best_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        epoch_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={elapsed:.1f}s [{epoch_time}]')

        # scheduler step
        if scheduler is not None:
            try:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            except Exception:
                pass

        # save best by val acc
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'classes': classes}, os.path.join(args.out, 'best.pth'))

        # early stopping by val_loss if requested
        if args.patience > 0:
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f'Early stopping: no improvement in val_loss for {args.patience} epochs')
                break


if __name__ == '__main__':
    main()
