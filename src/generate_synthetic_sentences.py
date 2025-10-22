"""Synthetic sentence generator moved into src/ for reorganization."""
# copy of original generate_synthetic_sentences.py kept here for clarity and use by scripts
import os, random, csv, argparse
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=500)
    p.add_argument('--out-dir', default='sentences_data_synth')
    p.add_argument('--ann', default='annotations/synthetic_annotations.csv')
    p.add_argument('--min_symbols', type=int, default=3)
    p.add_argument('--max_symbols', type=int, default=8)
    args = p.parse_args(argv)

    ROOT = os.path.abspath('.')
    SYMBOL_ROOT = os.path.join(ROOT, 'single_symbol_data')
    OUT = os.path.join(ROOT, args.out_dir)
    IMG_DIR = os.path.join(OUT, 'images')
    os.makedirs(IMG_DIR, exist_ok=True)

    classes = [d for d in os.listdir(SYMBOL_ROOT) if os.path.isdir(os.path.join(SYMBOL_ROOT, d))]
    print('Found classes:', len(classes))

    # build a pool of all symbol image paths (class, path) to maximize usage of available symbol images
    all_symbols = []
    for cls in classes:
        cls_dir = os.path.join(SYMBOL_ROOT, cls)
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_symbols.append((cls, os.path.join(cls_dir, f)))
    if len(all_symbols) == 0:
        raise RuntimeError('No symbol images found under data/ — ensure you have class subfolders with images')
    random.shuffle(all_symbols)

    # helper to pop next symbol from pool and reshuffle when exhausted
    def next_symbol():
        if len(all_symbols) == 0:
            return None
        item = all_symbols.pop()
        if len(all_symbols) == 0:
            # rebuild and reshuffle to reuse images
            for cls in classes:
                cls_dir = os.path.join(SYMBOL_ROOT, cls)
                for f in os.listdir(cls_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_symbols.append((cls, os.path.join(cls_dir, f)))
            random.shuffle(all_symbols)
        return item

    # track how often each class is used so we can bias sampling towards underused classes
    class_counts = {c: 0 for c in classes}

    def choose_class_least_used():
        # sample with probability inverse to current count (favor least used classes)
        weights = [1.0 / (1 + class_counts[c]) for c in classes]
        s = sum(weights)
        probs = [w / s for w in weights]
        return random.choices(classes, probs)[0]

    # track per-image usage to prefer unused files and maximize coverage
    per_image_used = {}
    for cls, path in all_symbols:
        per_image_used[path] = False

    w_canvas = 1024
    h_canvas = 128

    rows = []
    for i in range(args.count):
        symbols = random.randint(args.min_symbols, args.max_symbols)
        x = 8
        y_center = h_canvas // 2
        # subtle textured background: light gray base with speckle noise
        bg = Image.new('RGB', (w_canvas, h_canvas), (245,245,245))
        # add a denser speckle field with slight brightness variation
        speckles = int(w_canvas * h_canvas * 0.002)  # ~0.2% of pixels
        for _n in range(speckles):
            rx = random.randint(0, w_canvas-1); ry = random.randint(0, h_canvas-1)
            shade = random.randint(-10, 6)
            cur = bg.getpixel((rx, ry))
            new = tuple(max(0, min(255, c + shade)) for c in cur)
            bg.putpixel((rx, ry), new)
        # convert to RGBA so we can composite shadows and glyphs cleanly
        bg = bg.convert('RGBA')
        for j in range(symbols):
            # mostly pick from underused classes to maximize class coverage
            if random.random() < 0.8:
                cls = choose_class_least_used()
                cls_dir = os.path.join(SYMBOL_ROOT, cls)
                candidates = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not candidates:
                    continue
                img_name = random.choice(candidates)
                img_path = os.path.join(cls_dir, img_name)
            else:
                item = next_symbol()
                if item is None:
                    break
                cls, img_path = item
            class_counts[cls] = class_counts.get(cls, 0) + 1
            im = Image.open(img_path)
            # ensure we operate with RGBA; create alpha mask from luminance for RGB/JPEG inputs
            if im.mode != 'RGBA':
                # compute mask where ink is present (darker than near-white)
                gray = im.convert('L')
                # use a slightly higher threshold so faint strokes aren't lost (include pixels darker than 245)
                mask = gray.point(lambda p: 255 if p < 245 else 0)
                im = im.convert('RGBA')
                im.putalpha(mask)
            else:
                # if RGBA but alpha is empty, try to derive from luminance
                alpha = im.split()[-1]
                if alpha.getbbox() is None:
                    gray = im.convert('L')
                    # same slightly higher threshold for RGBA inputs with empty alpha
                    mask = gray.point(lambda p: 255 if p < 245 else 0)
                    im.putalpha(mask)
        # trim whitespace using alpha channel (tight crop)
            # crop tightly to alpha content, rotate, then recrop to remove any rotation padding
            try:
                alpha = im.split()[-1]
                bbox = alpha.getbbox()
                if bbox:
                    im = im.crop(bbox)
            except Exception:
                pass
            # random scale
            scale = random.uniform(0.8, 1.4)
            nw = max(12, int(im.width * scale))
            nh = max(12, int(im.height * scale))
            # if the symbol would be taller than the canvas, scale it down to fit
            max_h = max(8, h_canvas - 12)
            if nh > max_h:
                scale2 = max_h / float(nh)
                nw = max(12, int(nw * scale2))
                nh = max(12, int(nh * scale2))
            im = im.resize((nw, nh), resample=Image.BICUBIC)
            # optionally flip/rotate small
            if random.random() < 0.2:
                angle = random.uniform(-12, 12)
                im = im.rotate(angle, expand=True)
                # recrop after rotation to remove added transparent padding
                try:
                    alpha = im.split()[-1]
                    bbox = alpha.getbbox()
                    if bbox:
                        im = im.crop(bbox)
                except Exception:
                    pass
            # paste using alpha mask to avoid white rectangular frames and reduce halo
            # compute y and clamp so the glyph fits vertically in the canvas
            raw_y = y_center - im.height//2 + random.randint(-6,6)
            y = max(2, min(h_canvas - im.height - 2, raw_y))
            if im.mode == 'RGBA':
                mask = im.split()[-1]
                # very rarely erode the mask to remove white fringes; avoid doing this often because
                # it can remove thin/faint strokes
                if random.random() < 0.05:
                    try:
                        mask = mask.filter(ImageFilter.MinFilter(3))
                    except Exception:
                        pass
                # create a soft shadow by blurring the mask and pasting a dark translucent layer
                try:
                    # blur the mask for the shadow only so the shadow stays soft but
                    # doesn't occlude thin strokes too much
                    mask_blur = mask.filter(ImageFilter.GaussianBlur(radius=2))
                    # reduce shadow opacity to avoid darkening fine strokes
                    shadow = Image.new('RGBA', im.size, (0, 0, 0, 80))
                    bg.paste(shadow, (x + 2, y + 2), mask_blur)
                except Exception:
                    mask_blur = mask
                # optionally darken the glyph slightly to vary ink intensity
                try:
                    glyph = im.convert('RGBA')
                    # occasionally darken slightly but avoid strong darkening that removes detail
                    if random.random() < 0.35:
                        factor = random.uniform(0.86, 1.0)
                        # apply brightness on RGB channels only
                        rgb = glyph.convert('RGB')
                        rgb = ImageEnhance.Brightness(rgb).enhance(factor)
                        glyph = rgb.convert('RGBA')
                        glyph.putalpha(mask)
                    # paste glyph using the original mask to preserve thin/faint strokes
                    bg.paste(glyph, (x, y), mask)
                except Exception:
                    bg.paste(im, (x, y), mask)
            else:
                bg.paste(im, (x, y))
            x1 = x; y1 = y; x2 = x + im.width; y2 = y + im.height
            rows.append([os.path.join(args.out_dir, 'images', f'img_{i:05d}.png'), x1, y1, x2, y2, cls])
            # increase spacing between symbols to avoid crowding
            gap = int(im.width * random.uniform(0.95, 1.6)) + random.randint(12, 30)
            x += gap
            if x > w_canvas - 64:
                break
        # crop to used width
        used_w = min(w_canvas, x + 8)
        final = bg.crop((0,0,used_w,h_canvas))
        # apply mild global jitter: contrast, brightness, and tiny blur to blend composition
        try:
            # contrast and brightness jitter (narrower ranges to avoid washing out or over-darkening)
            final = final.convert('RGB')
            final = ImageEnhance.Contrast(final).enhance(random.uniform(0.98, 1.04))
            final = ImageEnhance.Brightness(final).enhance(random.uniform(0.99, 1.03))
            # optional tiny gaussian blur to mimic camera softness; keep very small
            blur_r = random.uniform(0.0, 0.45)
            if blur_r > 0.01:
                final = final.filter(ImageFilter.GaussianBlur(radius=blur_r))
        except Exception:
            final = final.convert('RGB')
        fname = os.path.join(IMG_DIR, f'img_{i:05d}.png')
        final.save(fname)

    # write CSV
    os.makedirs(os.path.dirname(args.ann), exist_ok=True)
    with open(args.ann, 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['image_path','x1','y1','x2','y2','label'])
        for r in rows:
            w.writerow(r)

    print('Wrote', args.count, 'synthetic images to', IMG_DIR)
    print('Wrote annotations to', args.ann)


if __name__ == '__main__':
    main()
