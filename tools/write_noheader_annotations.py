#!/usr/bin/env python3
"""Convert an annotations CSV to trainer-friendly no-header CSV.

Reads input CSV with header or without, allows normalizing slashes and stripping a leading path prefix.
Outputs CSV with rows: image_path,x1,y1,x2,y2,label (no header)
"""
import argparse
import csv
import os


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Input annotations CSV (may have header)')
    p.add_argument('--output', '-o', default='annotations/synthetic_annotations_noheader.csv', help='Output no-header CSV path')
    p.add_argument('--strip-prefix', default='', help='If provided, strip this leading path from image paths')
    p.add_argument('--normalize-slashes', action='store_true', help='Replace \\ with / in paths')
    p.add_argument('--skip-header', action='store_true', help='Force skipping first row as header')
    args = p.parse_args(argv)

    rows = []
    with open(args.input, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        first = next(reader)
        # decide if header
        has_header = True
        if args.skip_header:
            has_header = True
        else:
            hdr = [c.lower() for c in first]
            if len(hdr) >= 1 and ('image' in hdr[0] or 'path' in hdr[0] or 'image_path' in hdr[0]):
                has_header = True
            else:
                has_header = False
        if not has_header:
            # first is actually data
            rows.append(first)
        for r in reader:
            if len(r) < 6:
                continue
            rows.append(r)

    # normalize paths
    out_rows = []
    for r in rows:
        img = r[0]
        if args.normalize_slashes:
            img = img.replace('\\', '/')
        if args.strip_prefix:
            if img.startswith(args.strip_prefix):
                img = img[len(args.strip_prefix):]
                # remove leading slash if present
                if img.startswith('/') or img.startswith('\\'):
                    img = img[1:]
        out_rows.append([img, r[1], r[2], r[3], r[4], r[5]])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf8') as fo:
        w = csv.writer(fo)
        for r in out_rows:
            w.writerow(r)
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
