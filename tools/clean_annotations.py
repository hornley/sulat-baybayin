#!/usr/bin/env python3
"""
Check and optionally clean an annotations CSV by removing rows that reference
missing image files. Keeps the original annotations file intact and writes a
cleaned copy when requested.

Usage:
  python scripts\clean_annotations.py --data <images_dir> --ann <annotations.csv> [--remove] [--out cleaned.csv]

The script assumes CSV rows in the format used by the project:
  image_path,x1,y1,x2,y2,label
It will preserve a header row if present.
"""
import os
import csv
import argparse
from datetime import datetime


def resolve_image_path(img_path, data_root):
    """Resolve an image path referenced in the CSV against the provided data_root.
    Rules:
      - normalize the path
      - if the normalized path is absolute and exists, accept it
      - otherwise, try joining data_root + normalized path
      - if that doesn't exist, try stripping leading separators and join with data_root
      - finally, check the raw path as-is
    Returns the resolved absolute path if the file exists, else None.
    """
    if not img_path:
        return None
    img_norm = os.path.normpath(img_path)
    # Absolute and exists
    if os.path.isabs(img_norm) and os.path.exists(img_norm):
        return os.path.abspath(img_norm)

    # Try joining with data_root
    candidate = os.path.join(data_root, img_norm)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    # Strip any leading slashes/backslashes and try again (handles Windows leading-sep paths)
    stripped = img_norm.lstrip('\\/')
    if stripped != img_norm:
        candidate2 = os.path.join(data_root, stripped)
        if os.path.exists(candidate2):
            return os.path.abspath(candidate2)

    # Finally, check the raw path relative to cwd
    if os.path.exists(img_path):
        return os.path.abspath(img_path)

    return None


def has_header_row(row):
    # Expecting at least 6 columns: path,x1,y1,x2,y2,label
    if len(row) < 6:
        return False
    # Try parse x1 as float; if fails, assume header
    try:
        float(row[1])
        return False
    except Exception:
        return True


def clean_annotations(data_dir, ann_file, out_file=None, do_remove=False, max_list=50):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotations file not found: {ann_file}")

    missing = []
    kept = []
    # cache resolution results to avoid repeated filesystem checks for duplicate image paths
    resolved_cache = {}
    header = None

    with open(ann_file, 'r', encoding='utf8', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print('Annotation file is empty')
        return {'kept': 0, 'missing': 0, 'out': None}

    # Detect header
    first = rows[0]
    has_header_flag = has_header_row(first)
    idx = 0
    if has_header_flag:
        header = first
        idx = 1

    for r in rows[idx:]:
        if not r:
            continue
        img = r[0]
        if img in resolved_cache:
            resolved = resolved_cache[img]
        else:
            resolved = resolve_image_path(img, data_dir)
            resolved_cache[img] = resolved

        if resolved is None:
            missing.append((img, r))
        else:
            # keep original row; duplicates are preserved
            kept.append(r)

    total_rows = len(rows) - (1 if has_header_flag else 0)
    print(f"Checked annotations: {total_rows} rows")
    print(f"  Missing referenced image rows: {len(missing)}")
    # summarize unique missing image paths and counts
    missing_counts = {}
    for img, _ in missing:
        missing_counts[img] = missing_counts.get(img, 0) + 1
    unique_missing = len(missing_counts)
    if unique_missing > 0:
        print(f"  Unique missing image paths: {unique_missing}")
        print('\nSample missing entries (unique):')
        for i, (img, cnt) in enumerate(list(missing_counts.items())[:max_list]):
            print(f"  {i+1}. {img}  (rows: {cnt})")

    out_path = None
    if do_remove:
        # Determine out_file path
        if out_file:
            out_path = out_file
        else:
            base = os.path.splitext(os.path.basename(ann_file))[0]
            out_path = os.path.join(os.path.dirname(ann_file), f"{base}_cleaned.csv")
        
        # Write cleaned CSV preserving header if present
        with open(out_path, 'w', encoding='utf8', newline='') as wf:
            writer = csv.writer(wf)
            if header is not None:
                writer.writerow(header)
            for r in kept:
                writer.writerow(r)

        print(f"Wrote cleaned annotations to: {out_path}")
    else:
        print('No changes written (use --remove to write cleaned file)')

    return {'kept': len(kept), 'missing': len(missing), 'out': out_path}


def main():
    p = argparse.ArgumentParser(description='Check and optionally clean annotations CSV against image data dir')
    p.add_argument('--data', required=True, help='Path to images directory')
    p.add_argument('--ann', required=True, help='Path to annotations CSV')
    p.add_argument('--out', help='Path to write cleaned annotations (keeps original if not provided)')
    p.add_argument('--remove', action='store_true', help='Write cleaned annotation CSV without missing entries')
    p.add_argument('--max-list', type=int, default=50, help='Max missing entries to print')
    args = p.parse_args()

    res = clean_annotations(args.data, args.ann, out_file=args.out, do_remove=args.remove, max_list=args.max_list)
    print('\nSummary:')
    print(f"  Kept rows: {res['kept']}")
    print(f"  Missing rows: {res['missing']}")
    if res['out']:
        print(f"  Cleaned file: {res['out']}")


if __name__ == '__main__':
    main()
