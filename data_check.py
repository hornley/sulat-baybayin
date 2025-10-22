import os
from collections import Counter

def scan_data(root='single_symbol_data'):
    if not os.path.isdir(root):
        print(f'No data folder found at {root}')
        return
    counts = {}
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    for entry in sorted(os.listdir(root)):
        p = os.path.join(root, entry)
        if os.path.isdir(p):
            imgs = [f for f in os.listdir(p) if f.lower().endswith(exts)]
            counts[entry] = len(imgs)
    if not counts:
        print('No class subfolders with images found in data/')
        return
    total = sum(counts.values())
    print(f'Found {len(counts)} classes and {total} images')
    for k, v in sorted(counts.items()):
        note = ''
        if v < 2:
            note = '  <-- WARNING: <2 images (stratified split will fail)'
        elif v < 10:
            note = '  <-- few images (consider augmenting)'
        print(f'  {k}: {v}{note}')

if __name__ == '__main__':
    scan_data('data')
