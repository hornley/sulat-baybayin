import argparse
import torch
from PIL import Image, ImageDraw
from src.detection.dataset import BBoxDataset


def load_checkpoint(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    model_state = ck['model_state']
    classes = ck.get('classes', [])
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes) + 1)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model, classes


def draw_predictions(img_path, model, classes, device='cpu', thresh=0.5):
    img = Image.open(img_path).convert('RGB')
    import torchvision.transforms as T
    tensor = T.ToTensor()(img).to(device)
    with torch.no_grad():
        outputs = model([tensor])
    out = outputs[0]
    draw = ImageDraw.Draw(img)
    for box, label, score in zip(out['boxes'], out['labels'], out['scores']):
        if score < thresh:
            continue
        x1, y1, x2, y2 = map(float, box)
        cname = classes[label - 1] if label > 0 and label - 1 < len(classes) else str(int(label))
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.text((x1 + 3, y1 + 3), f"{cname}:{score:.2f}", fill='red')
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input', required=True, help='Image file or folder')
    parser.add_argument('--out', default='detections')
    parser.add_argument('--thresh', type=float, default=0.5)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, classes = load_checkpoint(args.ckpt, device=device)
    import os
    os.makedirs(args.out, exist_ok=True)
    paths = []
    if os.path.isdir(args.input):
        for f in os.listdir(args.input):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(args.input, f))
    else:
        paths = [args.input]
    for p in paths:
        out = draw_predictions(p, model, classes, device=device, thresh=args.thresh)
        out.save(os.path.join(args.out, os.path.basename(p)))
        print('wrote', os.path.join(args.out, os.path.basename(p)))


if __name__ == '__main__':
    main()
