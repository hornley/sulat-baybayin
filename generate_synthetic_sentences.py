"""Synthetic sentence generator moved into src/ for reorganization.

This version includes symbol normalization and optional on-disk caching of normalized
symbols so glyphs with varying sizes / bit-depths produce consistent results.
"""
# copy of original generate_synthetic_sentences.py kept here for clarity and use by scripts
import os, random, csv, argparse, hashlib
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
from src.shared.config_manager import generate_yaml_template, load_yaml_config, merge_configs, wait_for_user_edit


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


def normalize_symbol_from_image(img, target_height, bg_threshold_pct=99.0, min_size=8, resample=Image.BICUBIC, pad:int=0, mask_smooth_radius:float=1.0):
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
        try:
            r = max(0.0, float(mask_smooth_radius))
        except Exception:
            r = 1.0
        if r > 0.0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=r))
    else:
        mask = use_mask

    # 3) tight crop
    bbox = mask.getbbox()
    if not bbox:
        bbox = (0, 0, im.width, im.height)
    # expand bbox by pad pixels to avoid clipping thin outer edges
    try:
        p = int(max(0, pad))
    except Exception:
        p = 0
    if p > 0:
        x0, y0, x1, y1 = bbox
        x0 = max(0, x0 - p)
        y0 = max(0, y0 - p)
        x1 = min(im.width, x1 + p)
        y1 = min(im.height, y1 + p)
        bbox = (x0, y0, x1, y1)
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

    # Ensure RGB under fully transparent pixels is neutral (black) to avoid
    # bright-white RGB leaking through if intermediate processing converts
    # the layer to RGB before proper alpha compositing with the paper.
    try:
        alpha = rgba.split()[-1]
        # build a mask where alpha == 0
        trans_mask = alpha.point(lambda p: 255 if p == 0 else 0)
        if trans_mask.getbbox():
            black = Image.new('RGB', rgba.size, (0, 0, 0))
            rgb_only = rgba.convert('RGB')
            # composite: where trans_mask is 255 (fully transparent), pick black; otherwise keep rgb_only
            fixed_rgb = Image.composite(black, rgb_only, trans_mask)
            rgba = fixed_rgb.convert('RGBA')
            rgba.putalpha(alpha)
    except Exception:
        # if anything fails, return the original rgba to avoid breaking flow
        pass

    return rgba, final_mask


def _cache_path_for(img_path, cache_dir, target_height, bg_threshold_pct):
    # deterministic cache filename based on absolute path and params
    key = (os.path.abspath(img_path) + f"::h={target_height}|t={bg_threshold_pct}").encode('utf8')
    h = hashlib.md5(key).hexdigest()
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(cache_dir, f"{base}_{h}.png")


def _generate_random_ink_color():
    """Generate a random ink color suitable for writing.
    Returns RGB tuple with darker, more realistic colors.
    """
    # Predefined palette of realistic ink colors
    ink_palette = [
        (0, 0, 0),          # Black
        (20, 20, 20),       # Near black
        (40, 40, 100),      # Dark blue
        (50, 50, 150),      # Blue
        (60, 40, 120),      # Purple-blue
        (100, 60, 30),      # Brown
        (90, 50, 20),       # Dark brown
        (120, 30, 30),      # Dark red
        (100, 30, 50),      # Burgundy
        (50, 60, 50),       # Dark green
        (60, 60, 60),       # Dark gray
        (80, 70, 50),       # Sepia
        (30, 30, 80),       # Navy blue
        (70, 40, 40),       # Brown-red
    ]
    return random.choice(ink_palette)


def main(argv=None):
    p = argparse.ArgumentParser()
    
    # === YAML CONFIG OPTIONS ===
    p.add_argument('--args-input', default=None, help='Path to YAML config file. If not exists, will generate template and pause for user to edit')
    p.add_argument('--no-wait', action='store_true', help='Do not pause for user to edit generated YAML template (use defaults)')
    p.add_argument('--regen-args', action='store_true', help='Force regeneration of YAML template even if file exists')
    
    # === OUTPUT & GENERATION OPTIONS ===
    p.add_argument('--count', type=int, default=500, help='Number of images to generate')
    p.add_argument('--out-dir', default='sentences_data_synth', help='Output directory for generated images')
    p.add_argument('--ann', default=None, help='Path to annotation CSV (omit to skip writing annotations)')
    p.add_argument('--append', action='store_true', help='Continue numbering from highest existing image in --out-dir instead of starting at 0')
    p.add_argument('--min_symbols', type=int, default=3, help='Minimum symbols per sentence')
    p.add_argument('--max_symbols', type=int, default=8, help='Maximum symbols per sentence')
    
    # === SYMBOL NORMALIZATION ===
    p.add_argument('--symbol-height-frac', type=float, default=0.55, help='Fraction of canvas height to resize symbol to')
    p.add_argument('--bg-thresh-pct', type=float, default=99.5, help='Percentile used to decide background luminance cutoff')
    p.add_argument('--crop-pad', type=int, default=2, help='Extra pixels to pad around symbol bbox during crop to avoid edge clipping')
    p.add_argument('--mask-smooth-radius', type=float, default=0.8, help='Gaussian blur radius for mask smoothing during normalization (lower preserves thin strokes)')
    p.add_argument('--use-cache', action='store_true', help='Cache normalized symbols to disk for faster repeated runs')
    p.add_argument('--cache-dir', default='.symbol_cache', help='Directory to store normalized symbol cache')
    
    # === SHADOW & EROSION ===
    p.add_argument('--erode-shadow', action='store_true', help='Optionally erode mask before generating shadow (preserves glyph alpha)')
    p.add_argument('--erode-shadow-min-thickness', type=float, default=2.5, help='Minimum estimated stroke thickness (px) required to allow erosion for shadow')
    p.add_argument('--erode-shadow-prob', type=float, default=1.0, help='Probability (0..1) to apply shadow erosion when allowed')
    p.add_argument('--erode-glyph', action='store_true', help='Optionally erode glyph alpha mask (destructive) when stroke thickness >= threshold')
    p.add_argument('--erode-glyph-min-thickness', type=float, default=4.0, help='Minimum estimated stroke thickness (px) required to allow glyph erosion')
    p.add_argument('--erode-glyph-prob', type=float, default=1.0, help='Probability (0..1) to apply glyph erosion when allowed')
    
    # === INK APPEARANCE ===
    p.add_argument('--ink-color', type=str, default='black', help='Ink color: "original" (keep source colors), "grayscale", "black", "random" (random colors), or custom RGB as "R,G,B" (e.g. "50,50,150" for dark blue)')
    p.add_argument('--ink-color-variation', type=int, default=0, help='Random RGB variation per symbol (0-50). Each RGB channel varies by ±this amount. 0 disables variation.')
    p.add_argument('--ink-random-mode', type=str, choices=['per-symbol', 'per-image'], default='per-image', help='When --ink-color="random": "per-symbol" (each symbol different color) or "per-image" (all symbols in image same random color)')
    p.add_argument('--ink-random-prob', type=float, default=0.2, help='Probability (0.0-1.0) of applying random ink color when --ink-color="random". Allows mixing with fallback (black). Default: 0.2 (20%% of images/symbols get random color)')
    p.add_argument('--ink-darken-min', type=float, default=0.82, help='Minimum brightness factor for ink darkening (0..1, lower is darker)')
    p.add_argument('--ink-darken-max', type=float, default=0.96, help='Maximum brightness factor for ink darkening (<=1, lower is darker)')
    p.add_argument('--ink-alpha-gain', type=float, default=1.0, help='Multiply glyph alpha by this gain before paste (>=1.0 increases opacity)')
    p.add_argument('--ink-alpha-gamma', type=float, default=1.0, help='Gamma curve for glyph alpha before paste (<1.0 boosts mid alphas)')
    
    # === THIN STROKE HELPERS ===
    p.add_argument('--thin-stroke-thresh', type=float, default=3.5, help='Estimated stroke thickness below which extra boosts are applied')
    p.add_argument('--thin-alpha-gain', type=float, default=1.25, help='Additional alpha gain when stroke is thin (applied on top of ink-alpha-gain)')
    p.add_argument('--thin-alpha-gamma', type=float, default=0.85, help='Additional alpha gamma when stroke is thin')
    p.add_argument('--thin-darken-boost', type=float, default=0.08, help='Extra darkening for thin strokes (subtract from brightness factor; 0..0.3)')
    p.add_argument('--thin-alpha-floor', type=int, default=130, help='Minimum alpha for non-zero ink pixels when stroke is thin (0 keeps disabled)')
    
    # === PAPER TYPE & TEXTURE ===
    p.add_argument('--paper-type', choices=['white','yellow-paper','dotted'], default='white', help='Paper/background color and line style (white, yellow with lines, or dotted)')
    p.add_argument('--paper-type-mix', type=str, default=None, help='Comma-separated probabilities for [white,yellow-paper,dotted] e.g. "0.5,0.25,0.25" to randomly sample paper types per image')
    p.add_argument('--paper-texture', choices=['plain','grainy','crumpled'], default='plain', help='Paper texture/surface: plain (flat), grainy (speckled), or crumpled (warped)')
    p.add_argument('--paper-strength', type=float, default=None, help='Override paper blend strength (0..1). If omitted, per-type defaults are used')
    p.add_argument('--paper-yellow-strength', type=float, default=None, help='Optional: override yellow-paper strength (0..1). Higher values make yellow base stronger')
    
    # === RULED LINES (independent overlay) ===
    p.add_argument('--paper-lines-prob', type=float, default=0.0, help='Probability (0..1) to overlay ruled-paper lines on white/dotted paper (yellow-paper always shows lines regardless)')
    p.add_argument('--line-spacing', type=int, default=28, help='Pixel spacing between ruled lines (applies to all paper types)')
    p.add_argument('--line-opacity', type=int, default=40, help='Alpha opacity for ruled lines, 0-255 (applies to all paper types)')
    p.add_argument('--line-thickness', type=int, default=1, help='Line thickness in pixels (applies to all paper types)')
    p.add_argument('--line-jitter', type=int, default=2, help='Vertical jitter per line in pixels (applies to all paper types)')
    p.add_argument('--line-color', type=str, default='0,0,0', help='RGB color for lines as comma-separated ints, e.g. 0,0,0 (yellow-paper defaults to blue if black specified)')
    
    # === DOTTED PAPER ===
    p.add_argument('--dot-size', type=int, default=1, help='Radius for dotted paper dots')
    p.add_argument('--dot-opacity', type=int, default=50, help='Opacity for dotted paper dots (0-255)')
    p.add_argument('--dot-spacing', type=int, default=18, help='Spacing in pixels between dots (dotted paper always uses uniform grid)')
    p.add_argument('--dot-foreground', action='store_true', help='Also draw dots on top of the final image so they remain visible above ink')
    
    # === CRUMPLED TEXTURE ===
    p.add_argument('--crumple-strength', type=float, default=3, help='Scale factor for crumple warping (0..5+)')
    p.add_argument('--crumple-mesh-overlap', type=int, default=2, help='Pixel overlap between mesh tiles for crumpled paper (increase to 2-4 for high crumple strengths)')
    
    # === LIGHTING ===
    p.add_argument('--lighting', choices=['normal','bright','dim','shadows'], default='normal', help='Lighting variation to apply')
    p.add_argument('--brightness-jitter', type=float, default=0.03, help='Small random brightness jitter (used for normal mode)')
    p.add_argument('--contrast-jitter', type=float, default=0.03, help='Small random contrast jitter (used for normal mode)')
    p.add_argument('--shadow-intensity', type=float, default=0.0, help='Strength (0..1) of directional shadow overlay when lighting=shadows')
    
    args = p.parse_args(argv)

    # === YAML CONFIG LOADING ===
    if args.args_input is not None:
        yaml_path = args.args_input
        
        # Check if user wants to force regenerate the template
        if args.regen_args and os.path.exists(yaml_path):
            print(f'Regenerating YAML template at {yaml_path} due to --regen-args flag')
            os.remove(yaml_path)
        
        # If YAML file doesn't exist, generate template and optionally wait for user to edit
        if not os.path.exists(yaml_path):
            print(f'YAML config file not found at {yaml_path}')
            print('Generating template with current defaults...')
            
            # Extract all args except the YAML-specific ones
            yaml_args = {k: v for k, v in vars(args).items() 
                        if k not in ('args_input', 'no_wait', 'regen_args')}
            
            # Generate template
            generate_yaml_template(yaml_path, yaml_args)
            print(f'✓ Generated template: {yaml_path}')
            
            # Wait for user to edit unless --no-wait is specified
            if not args.no_wait:
                print()
                print('Please edit the YAML file to configure your parameters.')
                print('Press Enter when ready to continue...')
                wait_for_user_edit(yaml_path)
            else:
                print('Continuing with default values (--no-wait specified)')
        
        # Load YAML config and merge with CLI args (CLI takes precedence)
        print(f'Loading YAML config from {yaml_path}...')
        yaml_config = load_yaml_config(yaml_path)
        
        # Merge: YAML provides base values, CLI overrides
        merged = merge_configs(yaml_config, vars(args))
        
        # Update args namespace with merged values
        for key, value in merged.items():
            setattr(args, key, value)
        
        print('✓ YAML config loaded and merged with CLI arguments')

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

    # parse paper-type-mix if provided
    paper_type_choices = ['white', 'yellow-paper', 'dotted']
    paper_type_probs = None
    if args.paper_type_mix:
        try:
            probs = [float(x.strip()) for x in args.paper_type_mix.split(',')]
            if len(probs) == 3 and abs(sum(probs) - 1.0) < 0.01:
                paper_type_probs = probs
            else:
                print(f'Warning: --paper-type-mix must be 3 comma-separated floats summing to 1.0; ignoring')
        except Exception as e:
            print(f'Warning: failed to parse --paper-type-mix: {e}; ignoring')

    # determine starting index for image numbering
    start_idx = 0
    if args.append:
        try:
            existing = [f for f in os.listdir(IMG_DIR) if f.startswith('img_') and f.endswith('.png')]
            if existing:
                indices = []
                for fname in existing:
                    try:
                        num_part = fname.replace('img_', '').replace('.png', '')
                        indices.append(int(num_part))
                    except Exception:
                        pass
                if indices:
                    start_idx = max(indices) + 1
                    print(f'Appending: starting from img_{start_idx:05d}.png')
        except Exception as e:
            print(f'Warning: could not determine existing images for --append: {e}; starting from 0')

    rows = []
    # track if any thin strokes were detected in this image to avoid post-composition blur
    for i in range(start_idx, start_idx + args.count):
        # randomly choose paper type if mix is specified
        if paper_type_probs:
            current_paper_type = random.choices(paper_type_choices, weights=paper_type_probs)[0]
        else:
            current_paper_type = args.paper_type

        # Generate random ink color per-image if ink-color is "random" and mode is "per-image"
        per_image_random_color = None
        per_image_use_random = False
        try:
            if str(args.ink_color).strip().lower() == 'random' and str(args.ink_random_mode).strip().lower() == 'per-image':
                try:
                    prob = float(args.ink_random_prob)
                except Exception:
                    prob = 0.2
                if random.random() < prob:
                    per_image_random_color = _generate_random_ink_color()
                    per_image_use_random = True
                else:
                    per_image_random_color = None
                    per_image_use_random = False
        except Exception:
            per_image_random_color = None
            per_image_use_random = False

        symbols = random.randint(args.min_symbols, args.max_symbols)
        x = 8
        y_center = h_canvas // 2
        # artwork layer (transparent) where glyphs and shadows will be pasted
        art = Image.new('RGBA', (w_canvas, h_canvas), (255, 255, 255, 0))
        had_thin_strokes = False
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
                rgba, mask = normalize_symbol_from_image(
                    im,
                    target_h,
                    bg_threshold_pct=args.bg_thresh_pct,
                    pad=int(args.crop_pad),
                    mask_smooth_radius=float(args.mask_smooth_radius)
                )
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
                        # expand bbox by crop-pad to avoid clipping after rotation
                        try:
                            ppad = int(max(0, args.crop_pad))
                        except Exception:
                            ppad = 0
                        if ppad > 0:
                            x0, y0, x1, y1 = bbox
                            x0 = max(0, x0 - ppad)
                            y0 = max(0, y0 - ppad)
                            x1 = min(im.width, x1 + ppad)
                            y1 = min(im.height, y1 + ppad)
                            bbox = (x0, y0, x1, y1)
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
                    art.paste(shadow, (x + 2, y + 2), mask_blur)
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
                    # remember if we encountered thin strokes for this image
                    try:
                        if est_thickness < float(args.thin_stroke_thresh):
                            had_thin_strokes = True
                    except Exception:
                        pass

                    glyph = im.convert('RGBA')
                    # Consistently darken ink slightly so it reads above paper textures
                    try:
                        dmin = float(args.ink_darken_min)
                        dmax = float(args.ink_darken_max)
                        if dmin > dmax:
                            dmin, dmax = dmax, dmin
                        # clamp to sensible bounds
                        dmin = max(0.4, min(1.0, dmin))
                        dmax = max(0.4, min(1.0, dmax))
                        factor = random.uniform(dmin, dmax)
                        # push a bit darker if the stroke is thin
                        try:
                            thin_thresh = float(args.thin_stroke_thresh)
                            if est_thickness < thin_thresh:
                                factor = max(0.4, min(1.0, factor - float(args.thin_darken_boost)))
                        except Exception:
                            pass
                    except Exception:
                        factor = random.uniform(0.82, 0.96)
                    rgb = glyph.convert('RGB')
                    rgb = ImageEnhance.Brightness(rgb).enhance(factor)
                    # optionally neutralize/force ink color
                    try:
                        mode = str(args.ink_color).strip().lower()
                    except Exception:
                        mode = 'original'
                    
                    if mode == 'grayscale':
                        try:
                            gray = rgb.convert('L')
                            rgb = Image.merge('RGB', (gray, gray, gray))
                        except Exception:
                            pass
                    elif mode == 'black':
                        # set ink RGB to black; alpha carries stroke shape and edges
                        rgb = Image.new('RGB', rgb.size, (0, 0, 0))
                    elif mode == 'random':
                        # Use random ink color - either per-image or per-symbol
                        try:
                            random_mode = str(args.ink_random_mode).strip().lower()
                        except Exception:
                            random_mode = 'per-image'
                        
                        # Check probability for whether to apply random color
                        try:
                            prob = float(args.ink_random_prob)
                        except Exception:
                            prob = 0.2
                        
                        should_use_random = False
                        
                        if random_mode == 'per-image':
                            # For per-image mode, use the flag set at image level
                            if per_image_use_random and per_image_random_color is not None:
                                base_r, base_g, base_b = per_image_random_color
                                should_use_random = True
                        else:
                            # For per-symbol mode, check probability for each symbol
                            if random.random() < prob:
                                base_r, base_g, base_b = _generate_random_ink_color()
                                should_use_random = True
                        
                        if should_use_random:
                            # Apply per-symbol variation if specified
                            try:
                                variation = int(args.ink_color_variation)
                            except Exception:
                                variation = 0
                            
                            if variation > 0:
                                r = max(0, min(255, base_r + random.randint(-variation, variation)))
                                g = max(0, min(255, base_g + random.randint(-variation, variation)))
                                b = max(0, min(255, base_b + random.randint(-variation, variation)))
                            else:
                                r, g, b = base_r, base_g, base_b
                            
                            # Set ink to random color
                            rgb = Image.new('RGB', rgb.size, (r, g, b))
                        else:
                            # Fallback to black when probability doesn't trigger
                            rgb = Image.new('RGB', rgb.size, (0, 0, 0))
                    elif mode != 'original':
                        # try to parse as custom RGB value (e.g. "50,50,150")
                        try:
                            parts = [x.strip() for x in mode.split(',')]
                            if len(parts) == 3:
                                base_r = int(parts[0])
                                base_g = int(parts[1])
                                base_b = int(parts[2])
                                
                                # apply per-symbol color variation if specified
                                try:
                                    variation = int(args.ink_color_variation)
                                except Exception:
                                    variation = 0
                                
                                if variation > 0:
                                    # add random variation to each channel (±variation)
                                    r = max(0, min(255, base_r + random.randint(-variation, variation)))
                                    g = max(0, min(255, base_g + random.randint(-variation, variation)))
                                    b = max(0, min(255, base_b + random.randint(-variation, variation)))
                                else:
                                    r, g, b = base_r, base_g, base_b
                                
                                # clamp to valid range
                                r = max(0, min(255, r))
                                g = max(0, min(255, g))
                                b = max(0, min(255, b))
                                
                                # set ink to custom color
                                rgb = Image.new('RGB', rgb.size, (r, g, b))
                        except Exception:
                            # if parsing fails, default to black
                            rgb = Image.new('RGB', rgb.size, (0, 0, 0))
                    
                    glyph = rgb.convert('RGBA')

                    # determine which mask to use for glyph paste (may erode if user allows),
                    # and optionally boost alpha via gain/gamma to improve thin-stroke visibility
                    paste_mask = mask
                    if args.erode_glyph and est_thickness >= float(args.erode_glyph_min_thickness) and random.random() < float(args.erode_glyph_prob):
                        try:
                            paste_mask = mask.filter(ImageFilter.MinFilter(3))
                        except Exception:
                            paste_mask = mask

                    # apply alpha gain/gamma if requested (no dilation, preserves contours)
                    try:
                        gain = float(args.ink_alpha_gain)
                        gamma = float(args.ink_alpha_gamma)
                    except Exception:
                        gain, gamma = 1.0, 1.0
                    def _apply_gain_gamma(mask_img, g, gm):
                        try:
                            inv_gamma = (1.0 / gm) if gm > 0 else 1.0
                            lut = []
                            for a in range(256):
                                na = a / 255.0
                                na = pow(na, inv_gamma)
                                na = na * g
                                ia = int(max(0, min(255, round(na * 255.0))))
                                lut.append(ia)
                            return mask_img.point(lut)
                        except Exception:
                            return mask_img
                    if gain != 1.0 or gamma != 1.0:
                        paste_mask = _apply_gain_gamma(paste_mask, gain, gamma)
                    # for thin strokes, add a small extra boost
                    try:
                        thin_thresh = float(args.thin_stroke_thresh)
                        if est_thickness < thin_thresh:
                            tgain = float(args.thin_alpha_gain)
                            tgamma = float(args.thin_alpha_gamma)
                            if tgain != 1.0 or tgamma != 1.0:
                                paste_mask = _apply_gain_gamma(paste_mask, tgain, tgamma)
                            # optional: enforce a minimum alpha floor for any non-zero pixels
                            try:
                                alpha_floor = int(args.thin_alpha_floor)
                            except Exception:
                                alpha_floor = 0
                            if alpha_floor > 0:
                                try:
                                    lut = []
                                    for a in range(256):
                                        if a == 0:
                                            lut.append(0)
                                        else:
                                            lut.append(max(a, min(255, alpha_floor)))
                                    paste_mask = paste_mask.point(lut)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    glyph.putalpha(paste_mask)
                    art.paste(glyph, (x, y), paste_mask)
                except Exception:
                    art.paste(im, (x, y), mask)
            else:
                art.paste(im, (x, y))
            x1 = x; y1 = y; x2 = x + im.width; y2 = y + im.height
            rows.append([os.path.join(args.out_dir, 'images', f'img_{i:05d}.png'), x1, y1, x2, y2, cls])
            # increase spacing between symbols to avoid crowding
            gap = int(im.width * random.uniform(0.95, 1.6)) + random.randint(12, 30)
            x += gap
            if x > w_canvas - 64:
                break
        # crop to used width
        used_w = min(w_canvas, x + 8)
        art_cropped = art.crop((0,0,used_w,h_canvas))
    # apply mild global jitter: contrast, brightness, and tiny blur to blend composition
        try:
            # contrast and brightness jitter (narrower ranges to avoid washing out or over-darkening)
            # apply small per-image contrast/brightness to the art layer before composing with paper
            art_cropped = art_cropped.convert('RGBA')
            # preserve alpha: enhance RGB channels then reattach the original alpha
            try:
                alpha = art_cropped.split()[-1]
                rgb = art_cropped.convert('RGB')
                rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.98, 1.04))
                rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.99, 1.03))
                art_cropped = rgb.convert('RGBA')
                art_cropped.putalpha(alpha)
            except Exception:
                # fallback to previous behavior
                tmp = art_cropped.convert('RGB')
                tmp = ImageEnhance.Contrast(tmp).enhance(random.uniform(0.98, 1.04))
                tmp = ImageEnhance.Brightness(tmp).enhance(random.uniform(0.99, 1.03))
                art_cropped = tmp.convert('RGBA')
            # optional tiny gaussian blur to mimic camera softness; keep very small (avoid thinning strokes)
            # Skip blur entirely if this image contains thin strokes to preserve edge visibility
            blur_r = 0.0 if had_thin_strokes else random.uniform(0.0, 0.15)
            if blur_r > 0.01:
                art_cropped = art_cropped.filter(ImageFilter.GaussianBlur(radius=blur_r))
        except Exception:
            art_cropped = art_cropped.convert('RGBA')
        fname = os.path.join(IMG_DIR, f'img_{i:05d}.png')
        # optionally apply paper/background texture and lighting
        # ensure `final` has a sensible default in case augmentation fails
        final = art_cropped.convert('RGB')
        try:
            from src.shared.augmentations import overlay_paper_lines_pil, overlay_paper_texture_pil, apply_lighting_pil
            # apply paper texture first (this will include yellow-lined or dotted styles)
            try:
                color = tuple(int(x) for x in args.line_color.split(','))
                if len(color) != 3:
                    color = (0,0,0)
            except Exception:
                color = (0,0,0)

            # compose art (transparent) over a paper texture so paper shows through
            # dotted paper always uses uniform grid
            use_dot_uniform = True if current_paper_type == 'dotted' else False
            final = overlay_paper_texture_pil(
                art_cropped,
                paper_type=current_paper_type,
                paper_texture=args.paper_texture,
                line_color=color,
                line_opacity=int(args.line_opacity),
                line_spacing=int(args.line_spacing),
                line_thickness=int(args.line_thickness),
                line_jitter=int(args.line_jitter),
                paper_strength=(None if args.paper_strength is None else float(args.paper_strength)),
                paper_yellow_strength=(None if args.paper_yellow_strength is None else float(args.paper_yellow_strength)),
                crumple_strength=float(args.crumple_strength),
                crumple_mesh_overlap=int(args.crumple_mesh_overlap),
                    dot_size=int(args.dot_size),
                    dot_opacity_override=int(args.dot_opacity),
                    dot_uniform=use_dot_uniform,
                    dot_spacing=int(args.dot_spacing)
            )
            # optionally overlay ruled-paper lines independently (honor --paper-lines-prob)
            # note: yellow-paper always draws its ruled lines (ignoring --paper-lines-prob) but uses the line properties
            try:
                if float(args.paper_lines_prob) > 0.0 and random.random() < float(args.paper_lines_prob):
                    if current_paper_type != 'yellow-paper':
                        # overlay lines for white/dotted paper types when probability triggers
                        final = overlay_paper_lines_pil(final, line_color=color, line_opacity=int(args.line_opacity), line_spacing=int(args.line_spacing), line_thickness=int(args.line_thickness), jitter=int(args.line_jitter))
            except Exception:
                # if the overlay fails for any reason, continue with final as-is
                pass

            # apply lighting adjustments
            final = apply_lighting_pil(final, mode=args.lighting, brightness_jitter=float(args.brightness_jitter), contrast_jitter=float(args.contrast_jitter), shadow_intensity=float(args.shadow_intensity))

            # optionally draw dots on top of the final image so they remain visible above ink
            try:
                if args.dot_foreground and args.paper_type == 'dotted':
                    # draw dots onto an overlay and composite onto final
                    fg = Image.new('RGBA', final.size, (0,0,0,0))
                    draw_fg = ImageDraw.Draw(fg)
                    dot_r = max(1, int(args.dot_size))
                    dot_op = int(args.dot_opacity)
                    if args.dot_uniform:
                        sp = max(6, int(args.dot_spacing))
                        x0 = sp // 2
                        y0 = sp // 2
                        for yy in range(y0, final.height, sp):
                            for xx in range(x0, final.width, sp):
                                draw_fg.ellipse([(xx-dot_r, yy-dot_r), (xx+dot_r, yy+dot_r)], fill=(60,60,60,dot_op))
                    else:
                        sp = max(8, int(args.dot_spacing))
                        for yy in range(8, final.height, sp):
                            for xx in range(8, final.width, sp):
                                jx = xx + random.randint(-3,3)
                                jy = yy + random.randint(-3,3)
                                r = dot_r + random.randint(0,2)
                                draw_fg.ellipse([(jx-r, jy-r), (jx+r, jy+r)], fill=(70,70,70,dot_op))
                    final = Image.alpha_composite(final.convert('RGBA'), fg).convert('RGB')
            except Exception:
                pass

        except Exception as e:
            # keep the unaugmented art as final if something goes wrong; print error for debugging
            print('Warning: paper/lighting augmentation failed:', e)
            final = art_cropped.convert('RGB')
        final.save(fname)

    # determine annotation output path: if user didn't provide --ann, save inside the out-dir
    if args.ann is None:
        ann_path = os.path.join(OUT, 'annotations.csv')
    else:
        ann_path = args.ann

    os.makedirs(os.path.dirname(ann_path), exist_ok=True)
    
    # When using --append, append to existing annotations CSV instead of overwriting
    file_exists = os.path.exists(ann_path)
    if args.append and file_exists:
        # Append mode: don't write header, just add new rows
        with open(ann_path, 'a', newline='', encoding='utf8') as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r)
        print(f'Appended {len(rows)} annotations to', ann_path)
    else:
        # Write mode: create new file with header
        with open(ann_path, 'w', newline='', encoding='utf8') as f:
            w = csv.writer(f)
            w.writerow(['image_path','x1','y1','x2','y2','label'])
            for r in rows:
                w.writerow(r)
        print('Wrote annotations to', ann_path)

    print('Wrote', args.count, 'synthetic images to', IMG_DIR)


if __name__ == '__main__':
    main()
