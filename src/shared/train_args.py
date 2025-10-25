import argparse


def add_common_training_args(parser: argparse.ArgumentParser):
    """Add common training CLI args that both classification and detection trainers can reuse."""
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', '--batch-size', dest='batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader num_workers')
    parser.add_argument('--pin-memory', action='store_true', help='Use pin_memory in DataLoader')
    parser.add_argument('--prefetch-factor', type=int, default=None, help='DataLoader prefetch_factor (if supported)')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training (if CUDA available)')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--out', required=True, help='Output folder for checkpoints')
    parser.add_argument('--early-stop', type=int, default=0, help='Early stopping patience in epochs (0 to disable)')
    parser.add_argument('--device', default=None, help="Device override ('cpu' or 'cuda')")
    parser.add_argument('--resume-optimizer', action='store_true', help='When resuming from a checkpoint, also restore optimizer state if available')


def dataloader_kwargs_from_args(args):
    """Return a dict of DataLoader kwargs derived from parsed args.

    This safely includes `prefetch_factor` only when explicitly provided.
    """
    dl_kwargs = {
        'batch_size': getattr(args, 'batch', 32),
        'num_workers': getattr(args, 'num_workers', 0),
    }
    if getattr(args, 'pin_memory', False):
        dl_kwargs['pin_memory'] = True
    pf = getattr(args, 'prefetch_factor', None)
    if pf is not None:
        # include prefetch_factor; if running on older PyTorch this will raise at DataLoader construction time
        dl_kwargs['prefetch_factor'] = pf
    return dl_kwargs
