import argparse
import os
import time
import torch
import torchvision
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


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, scaler=None):
    model.train()
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
        if i % print_freq == 0:
            print(f'Epoch[{epoch}] Iter[{i}/{len(data_loader)}] Loss: {losses.item():.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Image root folder')
    parser.add_argument('--ann', required=True, help='CSV or COCO annotations file')
    add_common_training_args(parser)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--out', default='checkpoints/detection')
    parser.add_argument('--img-size', type=int, default=800, help='Resize shorter side to this value for detection')
    parser.add_argument('--resume', default=None, help='Checkpoint to resume from')
    parser.add_argument('--resume-optimizer', action='store_true', help='Also resume optimizer state from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl_kwargs = dataloader_kwargs_from_args(args)

    # build dataset and dataloader
    train_ds = BBoxDataset(args.data, ann_file=args.ann, transforms=None)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)

    classes = train_ds.classes
    model = get_model(len(classes) + 1)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

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

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler)
        save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'classes': classes}, os.path.join(args.out, f'epoch_{epoch}.pth'))


if __name__ == '__main__':
    main()
