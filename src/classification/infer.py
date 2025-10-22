import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from src.classification.model import BaybayinClassifier


def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    classes = ckpt.get('classes')
    model = BaybayinClassifier(num_classes=len(classes), weights=None)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model, classes


def predict_image(model, classes, img_path, device='cpu', topk=3, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        topk_probs, topk_idx = torch.topk(probs, k=min(topk, len(classes)))
    return [(classes[i], float(p)) for i, p in zip(topk_idx.tolist(), topk_probs.tolist())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint (best.pth)')
    parser.add_argument('--input', required=True, help='Image file or folder of images')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--img-size', type=int, default=224)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, classes = load_checkpoint(args.ckpt, device=device)

    paths = []
    if os.path.isdir(args.input):
        for fname in sorted(os.listdir(args.input)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                paths.append(os.path.join(args.input, fname))
    else:
        paths = [args.input]

    for p in paths:
        preds = predict_image(model, classes, p, device=device, topk=args.topk, img_size=args.img_size)
        print(f'File: {p}')
        for label, prob in preds:
            print(f'  {label}: {prob:.4f}')


if __name__ == '__main__':
    main()
