from PIL import Image, ImageDraw
import random
import numpy as np
from typing import Optional


def overlay_paper_lines_ndarray(img_arr, line_color=(0, 0, 0), line_opacity=30, line_spacing=24, line_thickness=1, jitter=2):
    """Overlay ruled paper lines onto an RGB numpy array image (H,W,3) and return uint8 array.

    This function is albumentations-friendly: it accepts and returns numpy arrays and does not alter bboxes.
    """
    if img_arr.dtype != np.uint8:
        img = Image.fromarray((img_arr * 255).astype('uint8'))
    else:
        img = Image.fromarray(img_arr)
    img = img.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    y = 0
    while y < h:
        jy = random.randint(-jitter, jitter)
        y_pos = y + jy
        draw.rectangle([(0, y_pos), (w, y_pos + line_thickness)], fill=(line_color[0], line_color[1], line_color[2], line_opacity))
        y += line_spacing
    combined = Image.alpha_composite(img, overlay)
    out = combined.convert('RGB')
    return np.array(out)


class AlbumentationsPaperLines:
    """Simple albumentations-like transform that overlays lines on numpy images.

    Usage with albumentations:
      A = Compose([..., Lambda(image=AlbumentationsPaperLines(...))])
    It does not alter bboxes.
    """
    def __init__(self, prob=0.3, line_color=(0,0,0), line_opacity=30, line_spacing=24, line_thickness=1, jitter=2):
        self.prob = prob
        self.line_color = line_color
        self.line_opacity = line_opacity
        self.line_spacing = line_spacing
        self.line_thickness = line_thickness
        self.jitter = jitter

    def __call__(self, img, **params):
        # img is numpy array HxWxC
        if random.random() < self.prob:
            return overlay_paper_lines_ndarray(img, line_color=self.line_color, line_opacity=self.line_opacity, line_spacing=self.line_spacing, line_thickness=self.line_thickness, jitter=self.jitter)
        return img



def overlay_paper_lines_pil(img, line_color=(0, 0, 0), line_opacity=30, line_spacing=24, line_thickness=1, jitter=2):
    """Overlay ruled paper lines onto a PIL RGB image.

    Args:
        img: PIL.Image (RGB)
        line_color: tuple RGB color
        line_opacity: int 0-255 alpha for line
        line_spacing: pixels between lines
        line_thickness: thickness in pixels
        jitter: max random vertical jitter per-line
    Returns:
        PIL.Image with lines overlaid.
    """
    if img.mode != 'RGBA':
        base = img.convert('RGBA')
    else:
        base = img.copy()
    overlay = Image.new('RGBA', base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size
    y = 0
    while y < h:
        jy = random.randint(-jitter, jitter)
        y_pos = y + jy
        draw.rectangle([(0, y_pos), (w, y_pos + line_thickness)], fill=(line_color[0], line_color[1], line_color[2], line_opacity))
        y += line_spacing
    combined = Image.alpha_composite(base, overlay)
    return combined.convert('RGB')


def paper_lines_transform_factory(prob=0.3, line_color=(0, 0, 0), line_opacity=30, line_spacing=24, line_thickness=1, jitter=2):
    """Return a callable transform(img) -> img that applies overlay with probability `prob`.
    Useful for passing into dataset transforms.
    """
    def _transform(img):
        if random.random() < prob:
            return overlay_paper_lines_pil(img, line_color=line_color, line_opacity=line_opacity, line_spacing=line_spacing, line_thickness=line_thickness, jitter=jitter)
        return img
    return _transform


def overlay_paper_texture_pil(img, paper_type='white', paper_texture='plain', line_color=(0,0,0), line_opacity=30, line_spacing=28, line_thickness=1, line_jitter=2,
                              paper_strength: Optional[float] = None, paper_yellow_strength: Optional[float] = None, crumple_strength: float = 1.0, crumple_mesh_overlap: int = 1, dot_size: int = 2, dot_opacity_override: Optional[int] = None,
                              dot_uniform: bool = False, dot_spacing: int = 16):
    """Apply a simple paper texture/background to a PIL image.

    paper_type: one of 'white', 'yellow-paper', 'dotted' (color and line style)
    paper_texture: one of 'plain', 'grainy', 'crumpled' (surface texture)
    Additional behavior: supports dotted grids, yellow lines, grain, and crumple warp.
    Optional parameters allow tuning of paper strength, crumple warp and dotted dot size/opacity.
    The function accepts legacy aliases 'yellow-lined' and 'yellow' and normalizes them to 'yellow-paper'.
    Legacy: if paper_type=='crumpled', it sets paper_texture='crumpled' for backward compatibility.
    """
    if img.mode != 'RGBA':
        base = img.convert('RGBA')
    else:
        base = img.copy()
    w, h = base.size

    # normalize legacy aliases so callers can use 'yellow' or 'yellow-lined' interchangeably
    try:
        if paper_type in ('yellow-lined', 'yellow'):
            paper_type = 'yellow-paper'
        # backward compatibility: if paper_type was 'crumpled', treat as white base + crumpled texture
        if paper_type == 'crumpled':
            paper_type = 'white'
            paper_texture = 'crumpled'
    except Exception:
        pass

    # start with a base paper color (paper_type controls color/lines)
    if paper_type == 'white':
        paper = Image.new('RGBA', (w, h), (250, 250, 245, 255))
    elif paper_type == 'yellow-paper':
        # more pronounced yellow tint for ruled paper so it reads as yellow under ink
        paper = Image.new('RGBA', (w, h), (240, 220, 100, 255))
    elif paper_type == 'dotted':
        paper = Image.new('RGBA', (w, h), (250, 250, 243, 255))
    else:
        paper = Image.new('RGBA', (w, h), (250, 250, 245, 255))

    # apply grain texture if requested (paper_texture controls surface)
    from PIL import ImageChops, ImageFilter
    if paper_texture == 'grainy':
        speck = Image.new('L', (w, h))
        px = speck.load()
        for yy in range(h):
            for xx in range(w):
                # low probability noise
                px[xx, yy] = int(max(0, min(255, 240 + random.randint(-8, 8))))
        speck = speck.filter(ImageFilter.GaussianBlur(radius=0.6))
        speck_rgb = Image.merge('RGBA', [speck.point(lambda p: p)]*3 + [Image.new('L', (w,h), 0)])
        paper = ImageChops.multiply(paper, speck_rgb)

    # dotted paper: draw dots now only if texture is NOT crumpled (so dots stay crisp)
    # If texture is crumpled, dots will be drawn after the warp below
    if paper_type == 'dotted' and paper_texture != 'crumpled':
        dot = Image.new('RGBA', (w, h), (0,0,0,0))
        draw = ImageDraw.Draw(dot)
        # spacing and radius behavior
        spacing = max(6, int(dot_spacing)) if dot_uniform else 16
        base_radius = max(1, int(dot_size))
        opacity = int(dot_opacity_override) if dot_opacity_override is not None else 200
        # make dots darker/more visible by default
        opacity = max(opacity, 120)
        if dot_uniform:
            # draw a strict grid with exact dot_size and no jitter
            x0 = spacing // 2
            y0 = spacing // 2
            for yy in range(y0, h, spacing):
                for xx in range(x0, w, spacing):
                    draw.ellipse([(xx-base_radius, yy-base_radius), (xx+base_radius, yy+base_radius)], fill=(90,90,90,opacity))
        else:
            # randomized grid with jitter and size variation
            for yy in range(8, h, spacing):
                for xx in range(8, w, spacing):
                    jitter_x = xx + random.randint(-3, 3)
                    jitter_y = yy + random.randint(-3, 3)
                    r = base_radius + random.randint(0, 3)  # slight size variation
                    draw.ellipse([(jitter_x-r, jitter_y-r), (jitter_x+r, jitter_y+r)], fill=(100,100,100,opacity))
        # slight blur to soften dots (simulate printed/dotted paper)
        try:
            # very small dots should not be blurred away
            blur_r = 0.0 if base_radius <= 1 else (0.4 if dot_uniform else 0.6)
            if blur_r > 0.0:
                dot = dot.filter(ImageFilter.GaussianBlur(radius=blur_r))
        except Exception:
            pass
        paper = Image.alpha_composite(paper, dot)

    # yellow-lined: overlay ruled lines now only if texture is NOT crumpled (so lines stay straight)
    # If texture is crumpled, lines will be drawn after the warp below
    if paper_type == 'yellow-paper' and paper_texture != 'crumpled':
        try:
            # choose a sensible default line color/opacity for yellow-lined paper if caller
            # left the defaults (black + low alpha). Use a blue rule color commonly
            # found on yellow-ruled paper so lines are visible.
            eff_color = line_color
            eff_opacity = line_opacity
            try:
                if (isinstance(line_color, tuple) and tuple(line_color) == (0,0,0)) or (isinstance(line_color, (list,)) and tuple(line_color) == (0,0,0)):
                    # caller used default black color; pick a blue-ish default for yellow-lined
                    eff_color = (30, 90, 160)
                if eff_opacity is None or int(eff_opacity) <= 40:
                    eff_opacity = max(120, int(eff_opacity) if eff_opacity is not None else 140)
            except Exception:
                eff_color = (30, 90, 160)
                eff_opacity = 140
            # reuse the shared overlay function to draw ruled lines on the paper image
            paper = overlay_paper_lines_pil(paper, line_color=eff_color, line_opacity=int(eff_opacity), line_spacing=line_spacing, line_thickness=line_thickness, jitter=line_jitter).convert('RGBA')
        except Exception:
            # fallback to manual drawing if something goes wrong
            lines = Image.new('RGBA', (w, h), (255,255,255,0))
            draw = ImageDraw.Draw(lines)
            y = 0
            while y < h:
                jy = random.randint(-line_jitter, line_jitter)
                y_pos = y + jy
                draw.rectangle([(0, y_pos), (w, y_pos + line_thickness)], fill=(line_color[0], line_color[1], line_color[2], line_opacity))
                y += line_spacing
            paper = Image.alpha_composite(paper, lines)

    # crumpled texture: create a bump map and apply a mesh-based warp for realistic folds
    if paper_texture == 'crumpled':
        # generate a noise-based bump map
        nm = Image.effect_noise((w, h), max(16, int(32 * crumple_strength))).convert('L')
        nm = nm.filter(ImageFilter.GaussianBlur(radius=2.0 * crumple_strength))

        # derive masks for highlights (raised) and shadows (recessed)
        try:
            mask_light = nm.point(lambda p: max(0, min(255, int((p - 128) * 2))))
            mask_dark = nm.point(lambda p: max(0, min(255, int((128 - p) * 2))))
        except Exception:
            mask_light = Image.new('L', (w, h), 0)
            mask_dark = Image.new('L', (w, h), 0)

        # brightness variants to composite as highlights and shadows
        from PIL import ImageEnhance
        light_amt = 1.25 + 0.18 * float(crumple_strength)
        dark_amt = max(0.35, 0.85 - 0.2 * float(crumple_strength))
        paper_rgb = paper.convert('RGB')
        light_img = ImageEnhance.Brightness(paper_rgb).enhance(light_amt)
        dark_img = ImageEnhance.Brightness(paper_rgb).enhance(dark_amt)

        try:
            tmp = Image.composite(light_img, paper_rgb, mask_light)
            paper_rgb = Image.composite(dark_img, tmp, mask_dark)
            paper = paper_rgb.convert('RGBA')
        except Exception:
            # fallback: subtly multiply to avoid complete darkness
            nm2 = nm.point(lambda p: int(max(160, min(255, p + int(20 * crumple_strength)))))
            nm_rgb = Image.merge('RGBA', [nm2]*3 + [Image.new('L', (w,h), 0)])
            paper = ImageChops.multiply(paper, nm_rgb)

        # mesh warp to produce geometric fold distortions
        try:
            # use a finer grid and larger offsets for stronger, sharper folds
            base_tile = int(max(12, 72 - int(30 * crumple_strength)))
            nx = max(4, int(w / base_tile))
            ny = max(4, int(h / base_tile))
            tile_w = int(w / nx)
            tile_h = int(h / ny)
            max_off = int(max(8, 18 * crumple_strength))
            mesh = []
            # overlap between tiles to avoid seams/holes (increase for high crumple strengths)
            overlap = max(1, int(crumple_mesh_overlap))
            for gy in range(ny):
                for gx in range(nx):
                    x0 = gx * tile_w
                    y0 = gy * tile_h
                    x1 = x0 + tile_w if gx < nx - 1 else w
                    y1 = y0 + tile_h if gy < ny - 1 else h
                    # slightly expand destination bbox to overlap neighbors
                    dx0 = max(0, x0 - overlap)
                    dy0 = max(0, y0 - overlap)
                    dx1 = min(w, x1 + overlap)
                    dy1 = min(h, y1 + overlap)
                    bbox = (dx0, dy0, dx1, dy1)
                    def jitter(px, py):
                        jx = px + random.uniform(-max_off, max_off)
                        jy = py + random.uniform(-max_off, max_off)
                        jx = max(0, min(w, jx))
                        jy = max(0, min(h, jy))
                        return (jx, jy)
                    p0 = jitter(x0, y0)
                    p1 = jitter(x1, y0)
                    p2 = jitter(x1, y1)
                    p3 = jitter(x0, y1)
                    quad = (p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
                    mesh.append((bbox, quad))
            # apply mesh warp. At high strength, MESH can leave uncovered pixels; use a fillcolor matching
            # the base paper to avoid visible black boxes, and clamp offsets to avoid degenerate quads.
            try:
                # blend max offset with tile size to avoid self-crossing quads
                max_tile_off = int(max(2, min(tile_w, tile_h) * 0.35))
                max_off = min(max_off, max_tile_off)
            except Exception:
                pass
            # choose a fillcolor from the current paper center to better match texture tone
            try:
                fc = paper.getpixel((min(w-1, w//2), min(h-1, h//2)))
                if len(fc) == 3:
                    fc = (fc[0], fc[1], fc[2], 255)
            except Exception:
                fc = (245, 244, 240, 255)
            try:
                paper = paper.transform((w, h), Image.MESH, mesh, resample=Image.BICUBIC, fillcolor=fc)
            except TypeError:
                # older Pillow without fillcolor support
                paper = paper.transform((w, h), Image.MESH, mesh, resample=Image.BICUBIC)
            except Exception:
                paper = paper.transform((w, h), Image.MESH, mesh, resample=Image.BICUBIC)

            # softly blend seams if any tiny gaps remain after warp
            try:
                from PIL import ImageFilter as _IF
                paper = paper.filter(_IF.GaussianBlur(radius=0.25))
            except Exception:
                pass

            # create a crease map from the bump map: edges emphasize fold creases
            try:
                edges = nm.filter(ImageFilter.FIND_EDGES).convert('L')
                # enhance edges and threshold to form thin crease lines
                edges = edges.point(lambda p: 255 if p > 40 else 0)
                crease = Image.new('RGBA', (w, h), (0,0,0,0))
                draw = ImageDraw.Draw(crease)
                # draw dark semi-transparent creases where edges are present
                crease_px = edges.load()
                for yy in range(h):
                    for xx in range(w):
                        if crease_px[xx, yy] > 0:
                            # draw a small darker spot to simulate a crease line
                            draw.point((xx, yy), fill=(20,20,20,160))
                paper = Image.alpha_composite(paper.convert('RGBA'), crease)
            except Exception:
                pass
        except Exception:
            pass

    # if dotted + crumpled, draw dots AFTER the warp so they stay crisp and unwarped
    if paper_type == 'dotted' and paper_texture == 'crumpled':
        dot = Image.new('RGBA', (w, h), (0,0,0,0))
        draw = ImageDraw.Draw(dot)
        spacing = max(6, int(dot_spacing)) if dot_uniform else 16
        base_radius = max(1, int(dot_size))
        opacity = int(dot_opacity_override) if dot_opacity_override is not None else 200
        opacity = max(opacity, 120)
        if dot_uniform:
            x0 = spacing // 2
            y0 = spacing // 2
            for yy in range(y0, h, spacing):
                for xx in range(x0, w, spacing):
                    draw.ellipse([(xx-base_radius, yy-base_radius), (xx+base_radius, yy+base_radius)], fill=(90,90,90,opacity))
        else:
            for yy in range(8, h, spacing):
                for xx in range(8, w, spacing):
                    jitter_x = xx + random.randint(-3, 3)
                    jitter_y = yy + random.randint(-3, 3)
                    r = base_radius + random.randint(0, 3)
                    draw.ellipse([(jitter_x-r, jitter_y-r), (jitter_x+r, jitter_y+r)], fill=(100,100,100,opacity))
        try:
            blur_r = 0.0 if base_radius <= 1 else (0.4 if dot_uniform else 0.6)
            if blur_r > 0.0:
                dot = dot.filter(ImageFilter.GaussianBlur(radius=blur_r))
        except Exception:
            pass
        paper = Image.alpha_composite(paper, dot)

    # if yellow-paper + crumpled, draw ruled lines AFTER the warp so they stay straight
    if paper_type == 'yellow-paper' and paper_texture == 'crumpled':
        try:
            eff_color = line_color
            eff_opacity = line_opacity
            try:
                if (isinstance(line_color, tuple) and tuple(line_color) == (0,0,0)) or (isinstance(line_color, (list,)) and tuple(line_color) == (0,0,0)):
                    eff_color = (30, 90, 160)
                if eff_opacity is None or int(eff_opacity) <= 40:
                    eff_opacity = max(120, int(eff_opacity) if eff_opacity is not None else 140)
            except Exception:
                eff_color = (30, 90, 160)
                eff_opacity = 140
            paper = overlay_paper_lines_pil(paper, line_color=eff_color, line_opacity=int(eff_opacity), line_spacing=line_spacing, line_thickness=line_thickness, jitter=line_jitter).convert('RGBA')
        except Exception:
            lines = Image.new('RGBA', (w, h), (255,255,255,0))
            draw = ImageDraw.Draw(lines)
            y = 0
            while y < h:
                jy = random.randint(-line_jitter, line_jitter)
                y_pos = y + jy
                draw.rectangle([(0, y_pos), (w, y_pos + line_thickness)], fill=(line_color[0], line_color[1], line_color[2], line_opacity))
                y += line_spacing
            paper = Image.alpha_composite(paper, lines)

    # Blend paper texture with the artwork. If the artwork has an alpha channel (transparent background),
    # composite the art over the paper so paper shows through. Otherwise, blend with a per-type alpha.
    try:
        paper_rgb = paper.convert('RGBA')
        base_rgba = base.convert('RGBA')

        # choose blend amount depending on paper type (subtle for white, stronger for crumpled)
        alpha_map = {
            'white': 0.12,
            'dotted': 0.36,
            'yellow-paper': 0.2,
            'crumpled': 0.68
        }
        alpha = alpha_map.get(paper_type, 0.2)
        if paper_strength is not None:
            alpha = float(max(0.0, min(1.0, paper_strength)))
        # allow a per-yellow override so callers can tune yellow-paper without affecting others
        if paper_type == 'yellow-paper' and paper_yellow_strength is not None:
            try:
                alpha = float(max(0.0, min(1.0, paper_yellow_strength)))
            except Exception:
                pass

        # resize paper to match
        if paper_rgb.size != base_rgba.size:
            paper_rgb = paper_rgb.resize(base_rgba.size, resample=Image.BICUBIC)

        if 'A' in base_rgba.getbands():
            # art has transparency: composite art over paper so paper shows through transparent areas
            composed = Image.alpha_composite(paper_rgb, base_rgba)
            # determine global blend factor per paper-type and texture
            if paper_texture == 'crumpled':
                blend_factor = float(max(0.45, min(1.0, alpha * 0.9)))
            elif paper_type == 'dotted':
                blend_factor = float(max(0.25, min(1.0, alpha * 0.6)))
            elif paper_type == 'yellow-paper':
                # reduce paper re-blend on yellow to avoid washing out ink (lower floor). If user explicitly set
                # paper_yellow_strength to 0, allow 0 re-blend (keep ink fully on top) to preserve very thin strokes.
                try:
                    if paper_yellow_strength is not None and float(paper_yellow_strength) == 0.0:
                        blend_factor = 0.0
                    else:
                        blend_factor = float(max(0.5, min(1.0, alpha * 0.5)))
                except Exception:
                    blend_factor = float(max(0.5, min(1.0, alpha * 0.5)))
            else:
                blend_factor = float(max(0.08, min(1.0, alpha * 0.25)))

            # Protect ink: reduce/disable paper re-blend where the art alpha is present
            ink_alpha = base_rgba.split()[-1]  # 0..255
            # Fully protect any pixel touched by ink from paper re-blend to avoid wash-out of thin strokes
            # Use a soft (blurred) mask from ink alpha to protect strokes without halos
            try:
                from PIL import ImageFilter as _IF
                protect_mask = ink_alpha.point(lambda a: 255 if a > 0 else 0)
                # slightly smaller blur keeps edges crisp while avoiding halos
                protect_mask = protect_mask.filter(_IF.GaussianBlur(radius=0.5))
            except Exception:
                protect_mask = ink_alpha.point(lambda a: 255 if a > 0 else 0)
            composed_rgb = composed.convert('RGB')
            paper_rgb_rgb = paper_rgb.convert('RGB')
            # first compute a uniformly blended candidate
            blended = Image.blend(composed_rgb, paper_rgb_rgb, alpha=blend_factor)
            # then composite: keep composed where ink is present; use blended elsewhere (paper-only areas)
            # This preserves dark ink above paper tint/dots.
            final = Image.composite(composed_rgb, blended, protect_mask)
        else:
            # no transparency: blend paper onto art
            final = Image.blend(base_rgba.convert('RGB'), paper_rgb.convert('RGB'), alpha=alpha)
    except Exception:
        final = base.convert('RGB')
    return final


def apply_lighting_pil(img, mode='normal', brightness_jitter=0.0, contrast_jitter=0.0, shadow_intensity=0.0):
    """Apply simple lighting variations to a PIL image.

    mode: 'normal'|'bright'|'dim'|'shadows'
    brightness_jitter, contrast_jitter: additive jitter ranges (e.g. 0.08)
    shadow_intensity: 0..1, amount of directional shadow when mode=='shadows'
    """
    from PIL import ImageEnhance, ImageChops
    import random

    out = img.convert('RGB')
    # base brightness/contrast adjustments
    if mode == 'bright':
        b_fac = random.uniform(1.06, 1.18)
        c_fac = random.uniform(1.02, 1.08)
    elif mode == 'dim':
        b_fac = random.uniform(0.66, 0.9)
        c_fac = random.uniform(0.9, 1.0)
    else:
        b_fac = 1.0 + random.uniform(-brightness_jitter, brightness_jitter)
        c_fac = 1.0 + random.uniform(-contrast_jitter, contrast_jitter)

    out = ImageEnhance.Brightness(out).enhance(b_fac)
    out = ImageEnhance.Contrast(out).enhance(c_fac)

    # shadows: apply a directional darkened overlay using a gradient mask
    if mode == 'shadows' or shadow_intensity > 0.0:
        w, h = out.size
        # create a linear gradient mask (left-to-right shadow)
        mask = Image.new('L', (w, h))
        mpx = mask.load()
        import math
        for yy in range(h):
            for xx in range(w):
                # gradient plus some vertical variation
                val = int(255 * (1.0 - shadow_intensity * (xx / float(max(1, w-1)))))
                mpx[xx, yy] = max(0, min(255, val + random.randint(-8, 8)))
        shadow = Image.new('RGBA', (w, h), (0,0,0, int(160 * shadow_intensity)))
        out = Image.composite(Image.alpha_composite(out.convert('RGBA'), shadow), out.convert('RGBA'), mask)
        out = out.convert('RGB')

    return out
