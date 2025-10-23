from PIL import Image, ImageDraw
import random


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
