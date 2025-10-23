"""Synthetic sentence generator moved into src/ for reorganization.

This version includes symbol normalization and optional on-disk caching of normalized
symbols so glyphs with varying sizes / bit-depths produce consistent results.
"""
# copy of original generate_synthetic_sentences.py kept here for clarity and use by scripts
import os, random, csv, argparse, hashlib
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


def _exif_transpose(img):
    # safeguard for orientation metadata
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def _percentile_threshold(gray_img, pct=99.0):
    """Compute a percentile-based threshold (0-255) from a PIL L image without numpy."""
    hist = gray_img.histogram()  # 256-length list
    total = sum(hist)
    if total == 0:
        return 255
    cutoff = int(total * (pct / 100.0))
    cumsum = 0
    for i, h in enumerate(hist):
        cumsum += h
        if cumsum >= cutoff:
            return i
    return 255


def normalize_symbol_from_image(img, target_height, bg_threshold_pct=99.0, min_size=8, resample=Image.BICUBIC):
    """Normalize a PIL image to an RGBA glyph of height target_height.

    Returns (rgba_img, mask)
    """
    # 1) normalize orientation and convert to RGBA 8-bit
    im = _exif_transpose(img).convert('RGBA')

    # 2) build mask: prefer existing alpha if present
    alpha = im.split()[-1]
    use_mask = None
    try:
        if alpha.getbbox() and alpha.getbbox() != (0, 0, im.width, im.height):
            use_mask = alpha
    except Exception:
        use_mask = None

    if use_mask is None:
        # derive mask from luminance with percentile adaptive threshold
        gray = im.convert('L')
        thr = _percentile_threshold(gray, pct=bg_threshold_pct)
        mask = gray.point(lambda p: 255 if p < thr else 0).convert('L')
        # smooth small speckles but keep faint strokes (avoid hard re-thresholding)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1))
    else:
        mask = use_mask

    # 3) tight crop
    bbox = mask.getbbox()
    if not bbox:
        bbox = (0, 0, im.width, im.height)
    cropped = im.crop(bbox)
    cropped_mask = mask.crop(bbox)

    # 4) resize to target height keeping aspect
    w, h = cropped.size
    if h == 0:
        h = 1
    scale = float(target_height) / float(h)
    new_w = max(min_size, int(round(w * scale)))
    new_h = max(min_size, int(round(h * scale)))
    resized = cropped.resize((new_w, new_h), resample=resample)
    resized_mask = cropped_mask.resize((new_w, new_h), resample=Image.NEAREST)

    # 5) keep mask as-is (do not erode) to preserve thin strokes
    final_mask = resized_mask

    # 6) recompose RGBA using mask as alpha
    rgb = resized.convert('RGB')
    rgba = rgb.copy()
    rgba.putalpha(final_mask)

    return rgba, final_mask


def _cache_path_for(img_path, cache_dir, target_height, bg_threshold_pct):
    # deterministic cache filename based on absolute path and params
    key = (os.path.abspath(img_path) + f"::h={target_height}|t={bg_threshold_pct}").encode('utf8')
    h = hashlib.md5(key).hexdigest()
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(cache_dir, f"{base}_{h}.png")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=500)
    p.add_argument('--out-dir', default='sentences_data_synth')
    p.add_argument('--ann', default=None, help='Path to annotation CSV (omit to skip writing annotations)')
    p.add_argument('--min_symbols', type=int, default=3)
    p.add_argument('--max_symbols', type=int, default=8)
    # normalization options
    p.add_argument('--symbol-height-frac', type=float, default=0.55, help='Fraction of canvas height to resize symbol to')
    p.add_argument('--use-cache', action='store_true', help='Cache normalized symbols to disk for faster repeated runs')
    p.add_argument('--cache-dir', default='.symbol_cache', help='Directory to store normalized symbol cache')
    p.add_argument('--bg-thresh-pct', type=float, default=99.0, help='Percentile used to decide background luminance cutoff')
    # erosion options
    p.add_argument('--erode-shadow', action='store_true', help='Optionally erode mask before generating shadow (preserves glyph alpha)')
    p.add_argument('--erode-shadow-min-thickness', type=float, default=2.5, help='Minimum estimated stroke thickness (px) required to allow erosion for shadow')
    p.add_argument('--erode-glyph', action='store_true', help='Optionally erode glyph alpha mask (destructive) when stroke thickness >= threshold')
    p.add_argument('--erode-glyph-min-thickness', type=float, default=4.0, help='Minimum estimated stroke thickness (px) required to allow glyph erosion')
    p.add_argument('--erode-shadow-prob', type=float, default=1.0, help='Probability (0..1) to apply shadow erosion when allowed')
    p.add_argument('--erode-glyph-prob', type=float, default=1.0, help='Probability (0..1) to apply glyph erosion when allowed')
    # paper line styling
    p.add_argument('--paper-lines-prob', type=float, default=0.0, help='Probability (0..1) to overlay ruled-paper lines on generated images')
    p.add_argument('--line-spacing', type=int, default=28, help='Pixel spacing between ruled lines')
    p.add_argument('--line-opacity', type=int, default=40, help='Alpha opacity for ruled lines (0-255)')
    p.add_argument('--line-thickness', type=int, default=1, help='Line thickness in pixels')
    p.add_argument('--line-jitter', type=int, default=2, help='Vertical jitter per line in pixels')
    p.add_argument('--line-color', type=str, default='0,0,0', help='RGB color for lines as comma-separated ints, e.g. 0,0,0')
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
            # load original image
            im = Image.open(img_path)

            # decide target symbol height from canvas
            target_h = max(8, int(h_canvas * args.symbol_height_frac))

            # optionally use cached normalized symbol
            cached = None
            if args.use_cache:
                os.makedirs(args.cache_dir, exist_ok=True)
                cp = _cache_path_for(img_path, args.cache_dir, target_h, args.bg_thresh_pct)
                if os.path.exists(cp):
                    try:
                        im = Image.open(cp).convert('RGBA')
                        # im is already normalized; mask is its alpha
                        mask = im.split()[-1]
                        cached = True
                    except Exception:
                        cached = False

            if not cached:
                rgba, mask = normalize_symbol_from_image(im, target_h, bg_threshold_pct=args.bg_thresh_pct)
                im = rgba
                if args.use_cache:
                    try:
                        cp = _cache_path_for(img_path, args.cache_dir, target_h, args.bg_thresh_pct)
                        im.convert('RGBA').save(cp)
                    except Exception:
                        pass
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
                # keep mask intact to preserve thin/faint strokes (no erosion)
                # create a soft shadow by blurring an eroded mask and pasting a dark translucent layer
                try:
                    # decide whether to erode the mask for shadow based on user flag
                    # and a simple stroke-thickness heuristic to avoid deleting thin strokes
                    shadow_mask = mask
                    if args.erode_shadow and random.random() < float(args.erode_shadow_prob):
                        try:
                            # estimate stroke thickness: use area / avg_dim heuristic
                            bbox = mask.getbbox()
                            if bbox:
                                area = float(sum(mask.histogram()[255:256]) if False else mask.convert('L').point(lambda p: 1 if p>0 else 0).histogram()[1])
                                # fallback heuristic: area pixels divided by average of width/height
                                w_m, h_m = mask.width, mask.height
                                avg_dim = max(1.0, (w_m + h_m) / 2.0)
                                est_thickness = max(0.0, float(area) / avg_dim)
                            else:
                                est_thickness = 0.0
                        except Exception:
                            est_thickness = 0.0
                        # apply erosion only if estimated thickness is above threshold
                        if est_thickness >= float(args.erode_shadow_min_thickness):
                            try:
                                shadow_mask = mask.filter(ImageFilter.MinFilter(3))
                            except Exception:
                                shadow_mask = mask
                    mask_blur = shadow_mask.filter(ImageFilter.GaussianBlur(radius=2))
                    # reduce shadow opacity to avoid darkening fine strokes
                    shadow = Image.new('RGBA', im.size, (0, 0, 0, 80))
                    bg.paste(shadow, (x + 2, y + 2), mask_blur)
                except Exception:
                    mask_blur = mask
                # optionally darken the glyph slightly to vary ink intensity
                try:
                    # compute a simple stroke-thickness estimate once and reuse
                    try:
                        bbox = mask.getbbox()
                        if bbox:
                            area = float(mask.convert('L').point(lambda p: 1 if p>0 else 0).histogram()[1])
                            w_m, h_m = mask.width, mask.height
                            avg_dim = max(1.0, (w_m + h_m) / 2.0)
                            est_thickness = max(0.0, float(area) / avg_dim)
                        else:
                            est_thickness = 0.0
                    except Exception:
                        est_thickness = 0.0

                    glyph = im.convert('RGBA')
                    # occasionally darken slightly but avoid strong darkening that removes detail
                    if random.random() < 0.35:
                        factor = random.uniform(0.86, 1.0)
                        # apply brightness on RGB channels only
                        rgb = glyph.convert('RGB')
                        rgb = ImageEnhance.Brightness(rgb).enhance(factor)
                        glyph = rgb.convert('RGBA')

                    # determine which mask to use for glyph paste (may erode if user allows)
                    paste_mask = mask
                    if args.erode_glyph and est_thickness >= float(args.erode_glyph_min_thickness) and random.random() < float(args.erode_glyph_prob):
                        try:
                            paste_mask = mask.filter(ImageFilter.MinFilter(3))
                        except Exception:
                            paste_mask = mask

                    glyph.putalpha(paste_mask)
                    bg.paste(glyph, (x, y), paste_mask)
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
        # optionally overlay paper lines for realism
        if args.paper_lines_prob and args.paper_lines_prob > 0.0:
            try:
                from src.shared.augmentations import overlay_paper_lines_pil
                if random.random() < float(args.paper_lines_prob):
                    # parse color
                    try:
                        color = tuple(int(x) for x in args.line_color.split(','))
                        if len(color) != 3:
                            color = (0,0,0)
                    except Exception:
                        color = (0,0,0)
                    final = overlay_paper_lines_pil(final, line_color=color, line_opacity=int(args.line_opacity), line_spacing=int(args.line_spacing), line_thickness=int(args.line_thickness), jitter=int(args.line_jitter))
            except Exception:
                pass
        final.save(fname)

    # determine annotation output path: if user didn't provide --ann, save inside the out-dir
    if args.ann is None:
        ann_path = os.path.join(OUT, 'annotations.csv')
    else:
        ann_path = args.ann

    os.makedirs(os.path.dirname(ann_path), exist_ok=True)
    with open(ann_path, 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['image_path','x1','y1','x2','y2','label'])
        for r in rows:
            w.writerow(r)
    print('Wrote annotations to', ann_path)

    print('Wrote', args.count, 'synthetic images to', IMG_DIR)


if __name__ == '__main__':
    main()
