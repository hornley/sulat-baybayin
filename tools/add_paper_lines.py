#!/usr/bin/env python3
"""
Add ruled paper lines to existing images.

This tool applies the same paper line styling from generate_synthetic_sentences.py
to input images, either a single file or all images in a directory.
"""
import argparse
import os
import sys
from pathlib import Path
from PIL import Image

# import the overlay function from shared augmentations
try:
    from src.shared.augmentations import overlay_paper_lines_pil
except ImportError:
    print("Error: Could not import overlay_paper_lines_pil from src.shared.augmentations")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def parse_color(color_str):
    """Parse comma-separated RGB string to tuple."""
    try:
        parts = [int(x.strip()) for x in color_str.split(',')]
        if len(parts) != 3:
            raise ValueError("Color must have exactly 3 components (R,G,B)")
        if not all(0 <= x <= 255 for x in parts):
            raise ValueError("Color components must be in range 0-255")
        return tuple(parts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid color format: {e}")


def process_image(input_path, output_path, line_color, line_opacity, line_spacing, line_thickness, line_jitter):
    """Apply paper lines to a single image."""
    try:
        # load image
        img = Image.open(input_path)
        
        # convert to RGB if needed
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
        
        # apply paper lines
        result = overlay_paper_lines_pil(
            img,
            line_color=line_color,
            line_opacity=line_opacity,
            line_spacing=line_spacing,
            line_thickness=line_thickness,
            jitter=line_jitter
        )
        
        # save result
        result.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Add ruled paper lines to images using the same styling from generate_synthetic_sentences.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add lines to a single image
  python add_paper_lines.py --input image.png --output image_lined.png
  
  # Process all images in a directory
  python add_paper_lines.py --input images/ --output images_lined/
  
  # Custom line styling
  python add_paper_lines.py --input image.png --output lined.png --line-spacing 30 --line-opacity 100 --line-color "0,0,255"
        """
    )
    
    # === INPUT/OUTPUT ===
    parser.add_argument('--input', required=True, help='Input image file or directory containing images')
    parser.add_argument('--output', required=True, help='Output image file or directory for processed images')
    
    # === PAPER LINE STYLING ===
    parser.add_argument('--line-spacing', type=int, default=28, help='Pixel spacing between ruled lines (default: 28)')
    parser.add_argument('--line-opacity', type=int, default=40, help='Alpha opacity for ruled lines, 0-255 (default: 40)')
    parser.add_argument('--line-thickness', type=int, default=1, help='Line thickness in pixels (default: 1)')
    parser.add_argument('--line-jitter', type=int, default=2, help='Vertical jitter per line in pixels (default: 2)')
    parser.add_argument('--line-color', type=str, default='0,0,0', help='RGB color for lines as comma-separated ints, e.g. "0,0,0" for black (default: 0,0,0)')
    
    args = parser.parse_args()
    
    # parse line color
    try:
        line_color = parse_color(args.line_color)
    except argparse.ArgumentTypeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # validate opacity range
    if not (0 <= args.line_opacity <= 255):
        print("Error: --line-opacity must be in range 0-255")
        sys.exit(1)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # check if input exists
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # determine if processing single file or directory
    if input_path.is_file():
        # single file mode
        print(f"Processing single image: {input_path}")
        
        # create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # process the image
        success = process_image(
            input_path,
            output_path,
            line_color,
            args.line_opacity,
            args.line_spacing,
            args.line_thickness,
            args.line_jitter
        )
        
        if success:
            print(f"✓ Saved result to: {output_path}")
        else:
            sys.exit(1)
    
    elif input_path.is_dir():
        # directory mode
        print(f"Processing directory: {input_path}")
        
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"Error: No image files found in {input_path}")
            sys.exit(1)
        
        print(f"Found {len(image_files)} images")
        
        # process each image
        success_count = 0
        for img_file in image_files:
            output_file = output_path / img_file.name
            print(f"  Processing: {img_file.name}...", end=' ')
            
            success = process_image(
                img_file,
                output_file,
                line_color,
                args.line_opacity,
                args.line_spacing,
                args.line_thickness,
                args.line_jitter
            )
            
            if success:
                print("✓")
                success_count += 1
            else:
                print("✗")
        
        print(f"\nCompleted: {success_count}/{len(image_files)} images processed successfully")
        
        if success_count < len(image_files):
            sys.exit(1)
    
    else:
        print(f"Error: Input path is neither a file nor a directory: {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
