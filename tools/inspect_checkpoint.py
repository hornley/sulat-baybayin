#!/usr/bin/env python3
"""
Inspect checkpoint files and display training details.

Displays information like:
- Epoch number
- Training metrics (loss, accuracy, mAP, etc.)
- Model configuration
- Class list
- Optimizer state
- Training arguments
"""

import argparse
import torch
import os
import json
from datetime import datetime


def format_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def inspect_checkpoint(ckpt_path, verbose=False, show_classes=False, show_args=False):
    """Inspect a checkpoint file and display its contents."""
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return
    
    # Get file info
    file_size = os.path.getsize(ckpt_path)
    file_mtime = datetime.fromtimestamp(os.path.getmtime(ckpt_path))
    
    print("=" * 70)
    print(f"📦 CHECKPOINT: {ckpt_path}")
    print("=" * 70)
    print(f"File size: {format_size(file_size)}")
    print(f"Modified: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load checkpoint
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # Determine checkpoint type
    is_detection = 'model_state' in ckpt and any('roi_heads' in k for k in ckpt.get('model_state', {}).keys())
    is_classification = 'model_state' in ckpt and any('fc' in k or 'classifier' in k for k in ckpt.get('model_state', {}).keys())
    
    ckpt_type = "Detection (Faster R-CNN)" if is_detection else "Classification" if is_classification else "Unknown"
    print(f"🔍 Type: {ckpt_type}")
    print()
    
    # Display main checkpoint keys
    print("📋 Checkpoint Contents:")
    print("-" * 70)
    for key in ckpt.keys():
        value = ckpt[key]
        if isinstance(value, dict):
            print(f"  • {key}: dict with {len(value)} keys")
        elif isinstance(value, list):
            print(f"  • {key}: list with {len(value)} items")
        elif isinstance(value, (int, float, str)):
            print(f"  • {key}: {value}")
        else:
            print(f"  • {key}: {type(value).__name__}")
    print()
    
    # Training Info
    print("🎯 Training Information:")
    print("-" * 70)
    
    if 'epoch' in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")
    
    if 'iteration' in ckpt or 'iter' in ckpt:
        iter_num = ckpt.get('iteration', ckpt.get('iter'))
        print(f"  Iteration: {iter_num}")
    
    if 'best_metric' in ckpt:
        print(f"  Best Metric: {ckpt['best_metric']:.6f}")
    
    if 'train_loss' in ckpt:
        print(f"  Training Loss: {ckpt['train_loss']:.6f}")
    
    if 'val_loss' in ckpt:
        print(f"  Validation Loss: {ckpt['val_loss']:.6f}")
    
    if 'val_acc' in ckpt:
        print(f"  Validation Accuracy: {ckpt['val_acc']:.4f}")
    
    if 'map' in ckpt:
        print(f"  mAP (mean Average Precision): {ckpt['map']:.4f}")
    
    if 'map_50' in ckpt:
        print(f"  mAP@0.5: {ckpt['map_50']:.4f}")
    
    if 'map_75' in ckpt:
        print(f"  mAP@0.75: {ckpt['map_75']:.4f}")
    
    print()
    
    # Classes
    if 'classes' in ckpt:
        classes = ckpt['classes']
        print(f"📚 Classes: {len(classes)} total")
        print("-" * 70)
        if show_classes:
            # Show all classes
            for i, cls in enumerate(classes):
                print(f"  {i+1:3d}. {cls}")
        else:
            # Show first 10 and last 10
            if len(classes) <= 20:
                for i, cls in enumerate(classes):
                    print(f"  {i+1:3d}. {cls}")
            else:
                for i in range(10):
                    print(f"  {i+1:3d}. {classes[i]}")
                print(f"  ... ({len(classes) - 20} more) ...")
                for i in range(len(classes) - 10, len(classes)):
                    print(f"  {i+1:3d}. {classes[i]}")
            print(f"\n  Use --show-classes to see all {len(classes)} classes")
        print()
    
    # Model State
    if 'model_state' in ckpt:
        model_state = ckpt['model_state']
        total_params = sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor))
        print(f"🧠 Model State:")
        print("-" * 70)
        print(f"  Total parameters: {total_params:,}")
        print(f"  State dict keys: {len(model_state)}")
        if verbose:
            print(f"\n  Layer shapes:")
            for i, (name, tensor) in enumerate(model_state.items()):
                if isinstance(tensor, torch.Tensor):
                    print(f"    {name}: {list(tensor.shape)}")
                if not verbose and i >= 10:
                    print(f"    ... ({len(model_state) - 10} more layers)")
                    break
        print()
    
    # Optimizer State
    if 'optimizer_state' in ckpt:
        print(f"⚙️  Optimizer State: Present")
        opt_state = ckpt['optimizer_state']
        if isinstance(opt_state, dict):
            if 'state' in opt_state:
                print(f"  Parameter groups: {len(opt_state.get('param_groups', []))}")
            if 'param_groups' in opt_state and len(opt_state['param_groups']) > 0:
                pg = opt_state['param_groups'][0]
                print(f"  Learning rate: {pg.get('lr', 'N/A')}")
                print(f"  Weight decay: {pg.get('weight_decay', 'N/A')}")
                print(f"  Momentum: {pg.get('momentum', 'N/A')}")
        print()
    
    # Scheduler State
    if 'scheduler_state' in ckpt:
        print(f"📅 LR Scheduler State: Present")
        sched_state = ckpt['scheduler_state']
        if isinstance(sched_state, dict):
            if 'last_epoch' in sched_state:
                print(f"  Last epoch: {sched_state['last_epoch']}")
            if '_step_count' in sched_state:
                print(f"  Step count: {sched_state['_step_count']}")
            if '_last_lr' in sched_state:
                print(f"  Last LR: {sched_state['_last_lr']}")
        print()
    
    # Training Arguments
    if 'args' in ckpt or 'train_args' in ckpt:
        args = ckpt.get('args', ckpt.get('train_args'))
        print(f"🔧 Training Arguments:")
        print("-" * 70)
        if show_args:
            if isinstance(args, dict):
                for key, value in sorted(args.items()):
                    print(f"  {key}: {value}")
            else:
                # argparse.Namespace
                for key, value in sorted(vars(args).items()):
                    print(f"  {key}: {value}")
        else:
            # Show key arguments only
            key_args = ['lr', 'batch_size', 'epochs', 'optimizer', 'momentum', 'weight_decay', 
                       'freeze_backbone', 'augment', 'data', 'stage']
            if isinstance(args, dict):
                for key in key_args:
                    if key in args:
                        print(f"  {key}: {args[key]}")
            else:
                for key in key_args:
                    if hasattr(args, key):
                        print(f"  {key}: {getattr(args, key)}")
            print(f"\n  Use --show-args to see all arguments")
        print()
    
    # Additional Info
    if 'timestamp' in ckpt:
        print(f"🕐 Timestamp: {ckpt['timestamp']}")
    
    if 'hostname' in ckpt:
        print(f"💻 Hostname: {ckpt['hostname']}")
    
    if 'git_commit' in ckpt:
        print(f"🔖 Git commit: {ckpt['git_commit']}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Inspect checkpoint files')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed model layer information')
    parser.add_argument('--show-classes', action='store_true', help='Show all classes (not just first/last 10)')
    parser.add_argument('--show-args', action='store_true', help='Show all training arguments')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()
    
    if args.json:
        # JSON output mode
        try:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            # Convert to JSON-serializable format
            info = {}
            for key, value in ckpt.items():
                if key in ['model_state', 'optimizer_state', 'scheduler_state']:
                    info[key] = f"<{type(value).__name__}>"
                elif isinstance(value, torch.Tensor):
                    info[key] = list(value.shape)
                elif isinstance(value, (int, float, str, bool, list)):
                    info[key] = value
                elif isinstance(value, dict):
                    info[key] = {k: (v if isinstance(v, (int, float, str, bool)) else str(type(v).__name__)) 
                                for k, v in value.items()}
                else:
                    info[key] = str(type(value).__name__)
            print(json.dumps(info, indent=2))
        except Exception as e:
            print(json.dumps({'error': str(e)}))
    else:
        # Human-readable output
        inspect_checkpoint(args.checkpoint, verbose=args.verbose, 
                         show_classes=args.show_classes, show_args=args.show_args)


if __name__ == '__main__':
    main()
