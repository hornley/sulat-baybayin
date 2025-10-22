#!/usr/bin/env python3
"""Two-stage training wrapper for classification.

Stage 1: train classifier head (freeze backbone)
Stage 2: resume from stage1 best checkpoint and fine-tune whole model

This wrapper calls the top-level `train.py` script in this repo and uses
`data_count.py` to print dataset statistics before training.

Defaults use AMP (if requested), pin_memory and 2 workers for DataLoader.
"""
import os
import sys
import argparse
import subprocess


def run_cmd(cmd, env=None):
    print('\n>> Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Root folder with class subfolders')
    p.add_argument('--out', default='checkpoints/two_stage', help='Output root for stage checkpoints')
    p.add_argument('--stage1-epochs', type=int, default=3)
    p.add_argument('--stage2-epochs', type=int, default=7)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-backbone-stage2', type=float, default=None, help='Optional backbone LR for stage2 (if unset, use lr*0.1)')
    p.add_argument('--amp', action='store_true', help='Enable mixed precision (if CUDA available)')
    p.add_argument('--device', default=None, help='Device override (cpu|cuda)')
    p.add_argument('--num-workers', type=int, default=2, help='DataLoader num_workers')
    p.add_argument('--pin-memory', action='store_true', help='Use DataLoader pin_memory')
    p.add_argument('--resume-optimizer', action='store_true', help='When resuming for stage2, restore optimizer state if available')
    p.add_argument('--batch-size', dest='batch_alias', type=int, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    python = sys.executable

    # 1) run data_count to show dataset composition
    try:
        run_cmd([python, os.path.join(os.path.dirname(__file__), '..', 'data_count.py'), '--root', args.data])
    except subprocess.CalledProcessError:
        print('data_count.py failed or printed warnings; continuing to training')

    # common training args
    common = [python, os.path.join(os.path.dirname(__file__), '..', 'train.py'), '--data', args.data,
              '--batch', str(args.batch), '--num-workers', str(args.num_workers)]
    if args.pin_memory:
        common.append('--pin-memory')
    if args.amp:
        common.append('--amp')
    if args.device:
        common += ['--device', args.device]

    # --- Stage 1: freeze backbone, train head ---
    stage1_out = os.path.join(out, 'stage1')
    os.makedirs(stage1_out, exist_ok=True)
    stage1_cmd = list(common) + ['--epochs', str(args.stage1_epochs), '--freeze-backbone', '--lr', str(args.lr), '--out', stage1_out]

    print('\n=== STAGE 1: training head (freeze backbone) ===')
    run_cmd(stage1_cmd)

    # find best checkpoint from stage1
    ckpt_stage1 = os.path.join(stage1_out, 'best.pth')
    if not os.path.exists(ckpt_stage1):
        print('Warning: stage1 best checkpoint not found at', ckpt_stage1)

    # --- Stage 2: resume and fine-tune whole network ---
    stage2_out = os.path.join(out, 'stage2')
    os.makedirs(stage2_out, exist_ok=True)
    lr_backbone = args.lr_backbone_stage2 if args.lr_backbone_stage2 is not None else args.lr * 0.1
    stage2_cmd = list(common) + ['--epochs', str(args.stage2_epochs), '--lr', str(args.lr), '--lr-backbone', str(lr_backbone), '--out', stage2_out]
    # resume from stage1 ckpt
    if os.path.exists(ckpt_stage1):
        stage2_cmd += ['--resume', ckpt_stage1]
        if args.resume_optimizer:
            stage2_cmd.append('--resume-optimizer')

    print('\n=== STAGE 2: fine-tune whole model (resume from stage1) ===')
    run_cmd(stage2_cmd)


if __name__ == '__main__':
    main()
