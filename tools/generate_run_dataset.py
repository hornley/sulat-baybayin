"""
Generate Run 11 Dataset
Runs 9 different synthetic data generation commands with varying paper types and textures.
Tracks progress and annotation counts throughout.
"""

import subprocess
import csv
import os
import sys
from datetime import datetime


def count_annotations(ann_file):
    """Count number of annotation rows in CSV (excluding header)."""
    if not os.path.exists(ann_file):
        return 0
    try:
        with open(ann_file, 'r', encoding='utf8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            return sum(1 for _ in reader)
    except Exception:
        return 0


def run_generation(name, count, paper_type, paper_texture, extra_args=""):
    """Run a single data generation command and report results."""
    
    # Build base command
    base_cmd = [
        'python', 'generate_synthetic_sentences.py',
        '--count', str(count),
        '--out-dir', 'sentences_data_synth/test',
        '--ann', 'sentences_data_synth/test/annotations.csv',
        '--append',
        '--bg-thresh-pct', '99.7',
        '--symbol-height-frac', '0.65',
        '--erode-shadow',
        '--erode-shadow-prob', '0',
        '--erode-glyph',
        '--erode-glyph-prob', '0',
        '--paper-type', paper_type,
        '--paper-texture', paper_texture,
        '--use-cache',
        '--ink-color', 'random',
        '--ink-random-mode', 'per-image',
        '--ink-random-prob', '0.5'
    ]
    
    # Add extra args
    if extra_args:
        base_cmd.extend(extra_args.split())
    
    # Get annotation count before
    ann_file = 'annotations/synthetic_annotations.csv'
    before_count = count_annotations(ann_file)
    
    # Run command
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"  Paper Type: {paper_type}")
    print(f"  Texture: {paper_texture}")
    print(f"  Count: {count}")
    print(f"{'='*70}")
    
    try:
        # Ensure all command parts are strings (avoid ints leaking in)
        base_cmd = [str(x) for x in base_cmd]
        # Prefer explicit Python executable for reproducibility
        base_cmd[0] = sys.executable if base_cmd and base_cmd[0] in ('python', 'python3') else base_cmd[0]
        result = subprocess.run(base_cmd, check=True, capture_output=True, text=True)
        
        # Get annotation count after
        after_count = count_annotations(ann_file)
        added_count = after_count - before_count
        
        # Report results
        print(f"\n✓ Generation complete!")
        print(f"  Images created: {count}")
        print(f"  Annotations added: {added_count}")
        print(f"  Total annotations: {after_count}")
        
        return {
            'name': name,
            'count': count,
            'added': added_count,
            'total': after_count,
            'success': True
        }
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Generation FAILED!")
        print(f"  Error: {e}")
        if e.stdout:
            print(f"  stdout: {e.stdout[-500:]}")  # Last 500 chars
        if e.stderr:
            print(f"  stderr: {e.stderr[-500:]}")
        
        return {
            'name': name,
            'count': count,
            'added': 0,
            'total': count_annotations('annotations/synthetic_annotations.csv'),
            'success': False
        }


def main():
    print("="*70)
    print("RUN 11 DATASET GENERATION")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define all 9 generation tasks
    tasks = [
        # WHITE PAPER
        {
            'name': 'White Paper - Plain',
            'count': 100,
            'paper_type': 'white',
            'paper_texture': 'plain',
            'extra_args': '--paper-lines-prob 0.5 --line-opacity 60 --lighting shadows --shadow-intensity 0.25 --ink-darken-min 0.86 --ink-darken-max 0.92'
        },
        {
            'name': 'White Paper - Crumpled',
            'count': 100,
            'paper_type': 'white',
            'paper_texture': 'crumpled',
            'extra_args': '--crumple-strength 3.5 --paper-lines-prob 0.5 --line-opacity 60 --lighting shadows --shadow-intensity 0.25 --ink-darken-min 0.86 --ink-darken-max 0.92'
        },
        {
            'name': 'White Paper - Grainy',
            'count': 100,
            'paper_type': 'white',
            'paper_texture': 'grainy',
            'extra_args': '--paper-lines-prob 0.5 --line-opacity 60 --lighting shadows --shadow-intensity 0.25 --ink-darken-min 0.86 --ink-darken-max 0.92'
        },
        
        # YELLOW PAPER
        {
            'name': 'Yellow Paper - Plain',
            'count': 100,
            'paper_type': 'yellow-paper',
            'paper_texture': 'plain',
            'extra_args': '--paper-yellow-strength 0 --ink-darken-min 0.86 --ink-darken-max 0.92 --ink-alpha-gain 1.5 --ink-alpha-gamma 0.78 --thin-stroke-thresh 4 --thin-alpha-gain 1.5 --thin-alpha-gamma 0.8 --thin-alpha-floor 160 --thin-darken-boost 0.15 --paper-strength 0.9 --ink-color black --crop-pad 3 --mask-smooth-radius 0.6'
        },
        {
            'name': 'Yellow Paper - Crumpled',
            'count': 100,
            'paper_type': 'yellow-paper',
            'paper_texture': 'crumpled',
            'extra_args': '--crumple-strength 3.5 --paper-yellow-strength 0 --ink-darken-min 0.86 --ink-darken-max 0.92 --ink-alpha-gain 1.5 --ink-alpha-gamma 0.78 --thin-stroke-thresh 4 --thin-alpha-gain 1.5 --thin-alpha-gamma 0.8 --thin-alpha-floor 160 --thin-darken-boost 0.15 --paper-strength 0.9 --ink-color black --crop-pad 3 --mask-smooth-radius 0.6'
        },
        {
            'name': 'Yellow Paper - Grainy',
            'count': 100,
            'paper_type': 'yellow-paper',
            'paper_texture': 'grainy',
            'extra_args': '--paper-yellow-strength 0 --ink-darken-min 0.86 --ink-darken-max 0.92 --ink-alpha-gain 1.5 --ink-alpha-gamma 0.78 --thin-stroke-thresh 4 --thin-alpha-gain 1.5 --thin-alpha-gamma 0.8 --thin-alpha-floor 160 --thin-darken-boost 0.15 --paper-strength 0.9 --ink-color black --crop-pad 3 --mask-smooth-radius 0.6'
        },
        
        # DOTTED PAPER
        {
            'name': 'Dotted Paper - Plain',
            'count': 100,
            'paper_type': 'dotted',
            'paper_texture': 'plain',
            'extra_args': '--lighting shadows --shadow-intensity 0.25 --ink-darken-min 0.86 --ink-darken-max 0.92 --thin-alpha-gain 1.4 --thin-darken-boost 0.12'
        },
        {
            'name': 'Dotted Paper - Crumpled',
            'count': 100,
            'paper_type': 'dotted',
            'paper_texture': 'crumpled',
            'extra_args': '--crumple-strength 3.5 --lighting shadows --shadow-intensity 0.25 --ink-darken-min 0.86 --ink-darken-max 0.92 --thin-alpha-gain 1.4 --thin-darken-boost 0.12'
        },
        {
            'name': 'Dotted Paper - Grainy',
            'count': 100,
            'paper_type': 'dotted',
            'paper_texture': 'grainy',
            'extra_args': '--lighting shadows --shadow-intensity 0.25 --ink-darken-min 0.86 --ink-darken-max 0.92 --thin-alpha-gain 1.4 --thin-darken-boost 0.12'
        },
    ]
    
    # Run all tasks
    results = []
    for task in tasks:
        result = run_generation(
            name=task['name'],
            count=task['count'],
            paper_type=task['paper_type'],
            paper_texture=task['paper_texture'],
            extra_args=task['extra_args']
        )
        results.append(result)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    total_requested = sum(r['count'] for r in results)
    total_added = sum(r['added'] for r in results)
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    
    print(f"Tasks completed: {successful}/{len(results)}")
    if failed > 0:
        print(f"Tasks failed: {failed}")
    print(f"\nTotal images requested: {total_requested}")
    print(f"Total annotations added: {total_added}")
    
    # Final annotation count
    ann_file = 'annotations/synthetic_annotations.csv'
    final_count = count_annotations(ann_file)
    print(f"Final annotation count: {final_count}")
    
    # Per-task breakdown
    print("\n" + "-"*70)
    print("PER-TASK BREAKDOWN")
    print("-"*70)
    print(f"{'Task':<30} {'Requested':<10} {'Added':<10} {'Status':<10}")
    print("-"*70)
    for r in results:
        status = "✓ OK" if r['success'] else "✗ FAILED"
        print(f"{r['name']:<30} {r['count']:<10} {r['added']:<10} {status:<10}")
    
    # Summary by paper type
    print("\n" + "-"*70)
    print("SUMMARY BY PAPER TYPE")
    print("-"*70)
    
    for paper_type in ['white', 'yellow-paper', 'dotted']:
        paper_results = [r for r in results if paper_type in r['name'].lower()]
        paper_requested = sum(r['count'] for r in paper_results)
        paper_added = sum(r['added'] for r in paper_results)
        display_name = paper_type.replace('-', ' ').title()
        print(f"{display_name:<20} Requested: {paper_requested:<6} Added: {paper_added:<6}")
    
    # Summary by texture
    print("\n" + "-"*70)
    print("SUMMARY BY TEXTURE")
    print("-"*70)
    
    for texture in ['plain', 'crumpled', 'grainy']:
        texture_results = [r for r in results if texture.lower() in r['name'].lower()]
        texture_requested = sum(r['count'] for r in texture_results)
        texture_added = sum(r['added'] for r in texture_results)
        print(f"{texture.title():<20} Requested: {texture_requested:<6} Added: {texture_added:<6}")
    
    print("\n" + "="*70)
    
    if failed > 0:
        print("\n⚠ Some tasks failed. Check output above for details.")
        return 1
    else:
        print("\n🎉 All tasks completed successfully!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
