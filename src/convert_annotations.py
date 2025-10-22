"""
Convert sentence-annotation CSV formats into the simple CSV format used by the detection loader.

This file was moved into `src/` during repository reorganization.
"""
import csv
import argparse
import os

KNOWN_INPUT_HEADERS = [
    ['label_name','bbox_x','bbox_y','bbox_width','bbox_height','image_name','image_width','image_height'],
    ['image_path','x1','y1','x2','y2','label']
]


def detect_format(header):
    h = [c.strip().lower() for c in header]
    if h == KNOWN_INPUT_HEADERS[0]:
        return 'sample_format'
    # basic target format detection
    if 'image_path' in h and 'label' in h:
        return 'target_format'
    # fallback: try to find columns
    if 'image_name' in h and 'bbox_x' in h:
        return 'sample_format'
    return None


def convert_sample_row(row, image_root):
    # expects dict with keys from sample_format header
    img = row.get('image_name') or row.get('image_path')
    x = float(row['bbox_x'])
    y = float(row['bbox_y'])
    w = float(row['bbox_width'])
    h = float(row['bbox_height'])
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    label = row['label_name'] if 'label_name' in row else row.get('label','')
    if not os.path.isabs(img) and image_root:
        img = os.path.join(image_root, img)
    return [img, str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), label]


def pass_through_row(row, header, image_root):
    # assume header contains image_path,x1,y1,x2,y2,label positions
    img = row.get('image_path') or row.get('image')
    if not os.path.isabs(img) and image_root:
        img = os.path.join(image_root, img)
    x1 = row.get('x1')
    y1 = row.get('y1')
    x2 = row.get('x2')
    y2 = row.get('y2')
    label = row.get('label')
    return [img, x1, y1, x2, y2, label]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Input CSV path')
    p.add_argument('--out', required=True, help='Output CSV path (target format)')
    p.add_argument('--image-root', default='', help='Optional image root folder to prepend to image paths')
    args = p.parse_args()

    with open(args.input, newline='', encoding='utf8') as inf:
        reader = csv.DictReader(inf)
        fmt = detect_format(reader.fieldnames)
        if fmt is None:
            print('Unrecognized input CSV header:', reader.fieldnames)
            return
        with open(args.out, 'w', newline='', encoding='utf8') as outf:
            writer = csv.writer(outf)
            writer.writerow(['image_path','x1','y1','x2','y2','label'])
            for row in reader:
                if fmt == 'sample_format':
                    out = convert_sample_row(row, args.image_root)
                else:
                    out = pass_through_row(row, reader.fieldnames, args.image_root)
                writer.writerow(out)
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
