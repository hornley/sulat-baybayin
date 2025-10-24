import os
import argparse

def main():
    p = argparse.ArgumentParser(description='Count images per-class in a folder-per-class dataset')
    p.add_argument('--root', default='single_symbol_data', help='Root folder containing class subfolders')
    p.add_argument('--exts', default='.png,.jpg,.jpeg,.bmp', help='Comma-separated image extensions to count')
    p.add_argument('--warn', type=int, default=2, help='Threshold for strong warning (default: 2)')
    p.add_argument('--few', type=int, default=10, help='Threshold for "few images" note (default: 10)')
    args = p.parse_args()

    exts = tuple(ext.strip().lower() for ext in args.exts.split(','))
    root = args.root

    if not os.path.isdir(root):
        print(f'No data folder found at: {root}')
        raise SystemExit(1)

    counts = {}
    total = 0
    for entry in sorted(os.listdir(root)):
        pth = os.path.join(root, entry)
        if os.path.isdir(pth):
            imgs = [f for f in os.listdir(pth) if f.lower().endswith(exts)]
            counts[entry] = len(imgs)
            total += len(imgs)

    print(f'Found {len(counts)} classes and {total} images in "{root}"')

    for k in sorted(counts):
        v = counts[k]
        note = ''
        if v < args.warn:
            note = f"  <-- WARNING: <{args.warn} images (stratified split may fail)"
        elif v < args.few:
            note = '  <-- few images (consider augmenting)'
        print(f'  {k}: {v}{note}')


if __name__ == '__main__':
    main()
