import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import make_dataloaders
from src.model import BaybayinClassifier
from src.utils import save_checkpoint, load_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
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
    parser.add_argument('--data', required=True, help='Root folder with class subfolders')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 weight decay for optimizer')
    parser.add_argument('--lr-backbone', type=float, default=None, help='Optional lower LR for backbone when fine-tuning')
    parser.add_argument('--lr-head', type=float, default=None, help='Optional LR for head/classifier when fine-tuning')
    parser.add_argument('--schedule', choices=['none', 'cosine', 'plateau'], default='none', help='LR schedule to use')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (by val_loss); set 0 to disable')
    # backward-compatible alias used previously by some scripts
    parser.add_argument('--early-stop', type=int, dest='patience', help=argparse.SUPPRESS)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--augment', action='store_true', help='Enable training data augmentation')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone layers and train only the classifier')
    parser.add_argument('--min-count', type=int, default=5, help='Minimum number of images required for a class to be included')
    parser.add_argument('--weights', type=str, default='default', help="Which pretrained weights to use: 'default', 'v1', or 'none'")
    parser.add_argument('--out', default='checkpoints')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, classes = make_dataloaders(args.data, batch_size=args.batch, img_size=args.img_size, augment=args.augment, min_count=args.min_count)

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
            # older torchvision: can't use enum API; set weights_arg to None so model uses random init
            weights_arg = None

    model = BaybayinClassifier(num_classes=len(classes), weights=weights_arg)
    model = model.to(device)

    if args.freeze_backbone:
        # Freeze all backbone parameters except the final fully-connected layer (backbone.fc)
        for name, param in model.backbone.named_parameters():
            if name.startswith('fc.'):
                continue
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    # build optimizer param groups (support discriminative LR for backbone/head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_groups = None
    # if user specified backbone/head LR, create groups
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
            # fallback: split by parameter shape/name unknown
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
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        print(f'Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={elapsed:.1f}s')

        # scheduler step
        if scheduler is not None:
            try:
                # ReduceLROnPlateau requires the val metric
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            except Exception:
                pass

        # save best by val acc
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'classes': classes}, os.path.join(args.out, 'best.pth'))

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
