import argparse
import torch
import torch.nn as nn
from dataset import make_dataloaders
from model import BaybayinClassifier
from collections import defaultdict


def evaluate_ckpt(ckpt_path, data_root='single_symbol_data', batch_size=16, img_size=224, min_count=5, device='cpu'):
    train_loader, val_loader, classes = make_dataloaders(data_root, batch_size=batch_size, img_size=img_size, min_count=min_count)
    if not classes:
        print('No classes found')
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BaybayinClassifier(num_classes=len(classes), weights=None)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    per_class = defaultdict(lambda: {'correct': 0, 'total': 0})
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            for p, t in zip(preds.tolist(), labels.tolist()):
                per_class[classes[t]]['total'] += 1
                if p == t:
                    per_class[classes[t]]['correct'] += 1
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    print(f'Val samples: {total}  Val loss: {total_loss/total:.4f}  Acc: {correct/total:.4f}')
    print('Per-class accuracy:')
    for cls in classes:
        info = per_class[cls]
        if info['total'] > 0:
            print(f'  {cls}: {info["correct"]}/{info["total"]} = {info["correct"]/info["total"]:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/best.pth')
    parser.add_argument('--data', default='single_symbol_data')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--min-count', type=int, default=5)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate_ckpt(args.ckpt, data_root=args.data, batch_size=args.batch, img_size=args.img_size, min_count=args.min_count, device=device)
