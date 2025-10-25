#!/usr/bin/env python3
"""Split annotations CSV (image_path,x1,y1,x2,y2,label) into train/val/test by image.

Saves files: <out_prefix>_train.csv, <out_prefix>_val.csv, <out_prefix>_test.csv
"""
import argparse
import csv
import os
import random
from collections import defaultdict


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Input annotations CSV (image_path,x1,y1,x2,y2,label)')
    p.add_argument('--out-prefix', '-o', default='annotations/annotations_real', help='Output prefix for train/val/test')
    p.add_argument('--val-percent', type=float, default=10.0, help='Percent of images to hold out for validation')
    p.add_argument('--test-percent', type=float, default=0.0, help='Percent of images to hold out for test')
    p.add_argument('--no-header', action='store_true', help='Input CSV has no header')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    args = p.parse_args(argv)

    random.seed(args.seed)

    images = defaultdict(list)
    with open(args.input, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        first = next(reader)
        # determine if header present
        has_header = True
        if args.no_header:
            # first row is data
            has_header = False
            # reset reader
            f.seek(0)
            reader = csv.reader(f)
        else:
            # check if first row looks like header by presence of 'image' or 'path'
            hdr = [c.lower() for c in first]
            if not any('image' in c or 'path' in c for c in hdr):
                # first row is actually data
                has_header = False
                f.seek(0)
                reader = csv.reader(f)
        for r in reader:
            if len(r) < 6:
                continue
            img = r[0]
            images[img].append(r)

    img_list = list(images.keys())
    random.shuffle(img_list)
    n = len(img_list)
    n_val = int(round((args.val_percent/100.0) * n))
    n_test = int(round((args.test_percent/100.0) * n))
    n_train = n - n_val - n_test
    train_imgs = set(img_list[:n_train])
    val_imgs = set(img_list[n_train:n_train+n_val])
    test_imgs = set(img_list[n_train+n_val:])

    def write_subset(prefix, subset_imgs):
        out_path = f"{prefix}.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', newline='', encoding='utf8') as fo:
            writer = csv.writer(fo)
            for img in subset_imgs:
                for row in images[img]:
                    writer.writerow(row)
        print('Wrote', out_path)

    write_subset(args.out_prefix + '_train', train_imgs)
    if n_val:
        write_subset(args.out_prefix + '_val', val_imgs)
    if n_test:
        write_subset(args.out_prefix + '_test', test_imgs)


if __name__ == '__main__':
    main()
