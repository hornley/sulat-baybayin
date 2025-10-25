import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from src.classification.model import BaybayinClassifier


def generate_output_dir(checkpoint_path, base_dir='classifications'):
    """
    Generate output directory based on checkpoint path.
    
    Example:
        checkpoints/classification/colab_run10/stage2/best.pth
        -> classifications/colab_run10/stage2/infer1
        
        checkpoints/single_symbol/best.pth
        -> classifications/single_symbol/infer1
    """
    # Parse checkpoint path to extract run identifier and stage
    ckpt_parts = os.path.normpath(checkpoint_path).split(os.sep)
    
    # Find 'checkpoints' in path and extract everything after it (excluding the filename)
    try:
        ckpt_idx = ckpt_parts.index('checkpoints')
        # Get path components after 'checkpoints' but before filename
        run_parts = ckpt_parts[ckpt_idx + 1:-1]  # Exclude 'checkpoints' and filename
        # Remove 'detection' or 'classification' from run_parts if present
        run_parts = [p for p in run_parts if p not in ('detection', 'classification')]
    except (ValueError, IndexError):
        # If 'checkpoints' not in path, use parent directory of checkpoint file
        run_parts = [os.path.basename(os.path.dirname(checkpoint_path))]
    
    # Build base path for this run
    if run_parts:
        run_path = os.path.join(base_dir, *run_parts)
    else:
        run_path = base_dir
    
    # Find next available infer number
    os.makedirs(run_path, exist_ok=True)
    existing = [d for d in os.listdir(run_path) if d.startswith('infer') and os.path.isdir(os.path.join(run_path, d))]
    
    # Extract numbers from existing infer folders
    numbers = []
    for d in existing:
        try:
            num = int(d.replace('infer', ''))
            numbers.append(num)
        except ValueError:
            continue
    
    next_num = max(numbers) + 1 if numbers else 1
    return os.path.join(run_path, f'infer{next_num}')


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
    parser.add_argument('--output', default=None, help='Output folder for per-image predictions (auto-generated if not specified)')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--compile-inferred', action='store_true', help='Write a compiled CSV/text of all predictions')
    args = parser.parse_args()

    # Auto-generate output directory if not provided
    if args.output is None:
        args.output = generate_output_dir(args.ckpt, base_dir='classifications')
        print(f'Auto-generated output directory: {args.output}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, classes = load_checkpoint(args.ckpt, device=device)

    paths = []
    if os.path.isdir(args.input):
        for fname in sorted(os.listdir(args.input)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                paths.append(os.path.join(args.input, fname))
    else:
        paths = [args.input]

    os.makedirs(args.output, exist_ok=True)

    compiled_rows = []

    for p in paths:
        preds = predict_image(model, classes, p, device=device, topk=args.topk, img_size=args.img_size)
        # write per-image predictions to a text file: label <tab> confidence
        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(args.output, f"{base}.txt")
        with open(out_path, 'w', encoding='utf8') as f:
            for label, prob in preds:
                f.write(f"{label}\t{prob:.6f}\n")
        print(f'Wrote predictions for {p} -> {out_path}')

        if args.compile_inferred:
            # flattened row: image_path, label1, prob1, label2, prob2, ...
            row = [p]
            for label, prob in preds:
                row.append(label)
                row.append(f"{prob:.6f}")
            compiled_rows.append(row)

    # write compiled outputs if requested
    if args.compile_inferred and compiled_rows:
        # write a simple TXT with top-1 label per line and a CSV with expanded top-k
        txt_out = os.path.join(args.output, 'compiled_inferred.txt')
        csv_out = os.path.join(args.output, 'compiled_predictions.csv')
        with open(txt_out, 'w', encoding='utf8') as tf:
            for r in compiled_rows:
                # r[0]=image_path, r[1]=label1
                tf.write(f"{r[0]}\t{r[1]}\t{r[2]}\n")
        import csv
        with open(csv_out, 'w', newline='', encoding='utf8') as cf:
            w = csv.writer(cf)
            # header
            header = ['image_path']
            for k in range(1, args.topk + 1):
                header += [f'label_{k}', f'prob_{k}']
            w.writerow(header)
            for r in compiled_rows:
                w.writerow(r)
        print(f'Wrote compiled_inferred: {txt_out} and CSV: {csv_out}')


if __name__ == '__main__':
    main()
