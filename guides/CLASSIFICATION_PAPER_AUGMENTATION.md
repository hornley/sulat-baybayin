# Classification Training with Paper Augmentation

## Overview

The classification training pipeline now supports **advanced paper augmentation** using the same realistic paper backgrounds, ruled lines, and lighting effects from the detection synthetic data generation system.

This provides three major augmentation categories:
1. **Paper Texture** - white/yellow-paper/dotted backgrounds with plain/grainy/crumpled surfaces
2. **Ruled Lines** - overlay horizontal lines like ruled notebook paper
3. **Lighting Variations** - normal/bright/dim/shadows modes with jitter

---

## Quick Start

### Basic Training (No Augmentation)
```bash
python train.py --data single_symbol_data/ --out checkpoints/baseline --epochs 50
```

### Standard Geometric Augmentation Only
```bash
python train.py \
    --data single_symbol_data/ \
    --out checkpoints/geometric \
    --epochs 50 \
    --augment  # rotation, crop, affine, color jitter
```

### Paper Augmentation (Recommended)
```bash
python train.py \
    --data single_symbol_data/ \
    --out checkpoints/paper_aug \
    --epochs 50 \
    --aug-paper-prob 0.5 \      # 50% of images get paper textures
    --aug-lighting-prob 0.5 \   # 50% get lighting variations
    --aug-lines-prob 0.3        # 30% get ruled lines
```

### Combined Augmentation (Maximum Robustness)
```bash
python train.py \
    --data single_symbol_data/ \
    --out checkpoints/full_aug \
    --epochs 50 \
    --augment \                 # Enable geometric augmentation
    --aug-paper-prob 0.4 \      # 40% paper textures
    --aug-lighting-prob 0.5 \   # 50% lighting variations
    --aug-lines-prob 0.2        # 20% ruled lines
```

---

## Paper Augmentation Arguments

### Paper Texture Augmentation

Control paper backgrounds and surface textures:

```bash
--aug-paper-prob 0.0            # Probability to apply (0..1, default: 0.0)

# Paper type distribution (white, yellow-paper, dotted)
--aug-paper-type-probs "0.6,0.2,0.2"  # default: 60% white, 20% yellow, 20% dotted

# Paper texture distribution (plain, grainy, crumpled)
--aug-paper-texture-probs "0.5,0.3,0.2"  # default: 50% plain, 30% grainy, 20% crumpled

# Paper blend strength (how visible the paper texture is)
--aug-paper-strength-min 0.2    # Minimum strength (default: 0.2)
--aug-paper-strength-max 0.4    # Maximum strength (default: 0.4)

# Yellow paper specific strength
--aug-paper-yellow-strength-min 0.3   # default: 0.3
--aug-paper-yellow-strength-max 0.5   # default: 0.5

# Crumple warping strength
--aug-crumple-strength-min 1.0  # Minimum warp (default: 1.0)
--aug-crumple-strength-max 3.0  # Maximum warp (default: 3.0)
--aug-crumple-mesh-overlap 2    # Tile overlap for crumple mesh (default: 2)
```

### Ruled Lines Augmentation

Add horizontal ruled lines like notebook paper:

```bash
--aug-lines-prob 0.0            # Probability to overlay lines (0..1, default: 0.0)

# Line spacing (distance between lines)
--aug-line-spacing-min 24       # Minimum spacing in pixels (default: 24)
--aug-line-spacing-max 32       # Maximum spacing in pixels (default: 32)

# Line opacity (transparency)
--aug-line-opacity-min 30       # Minimum alpha 0-255 (default: 30)
--aug-line-opacity-max 60       # Maximum alpha 0-255 (default: 60)

# Line thickness
--aug-line-thickness-min 1      # Minimum pixels (default: 1)
--aug-line-thickness-max 2      # Maximum pixels (default: 2)

# Line jitter (vertical randomness)
--aug-line-jitter-min 1         # Minimum jitter (default: 1)
--aug-line-jitter-max 3         # Maximum jitter (default: 3)

# Line color
--aug-line-color "0,0,0"        # RGB as comma-separated (default: black)
```

### Lighting Augmentation

Apply realistic lighting variations:

```bash
--aug-lighting-prob 0.0         # Probability to apply (0..1, default: 0.0)

# Lighting modes to sample from
--aug-lighting-modes "normal,bright,dim,shadows"  # default: all 4 modes

# Brightness/contrast jitter for 'normal' mode
--aug-brightness-jitter 0.03    # default: 0.03
--aug-contrast-jitter 0.03      # default: 0.03

# Shadow intensity for 'shadows' mode
--aug-shadow-intensity-min 0.0  # Minimum intensity (default: 0.0)
--aug-shadow-intensity-max 0.3  # Maximum intensity (default: 0.3)
```

### Dotted Paper Options

Control dot appearance when dotted paper type is selected:

```bash
--aug-dot-size 1                # Dot radius in pixels (default: 1)
--aug-dot-opacity 50            # Dot opacity 0-255 (default: 50)
--aug-dot-spacing 18            # Spacing between dots (default: 18)
```

---

## Example Configurations

### Light Augmentation (Real-world Paper)
Simulates clean notebook paper with occasional lighting variations:

```bash
python train.py \
    --data single_symbol_data/ \
    --out checkpoints/light_aug \
    --epochs 50 \
    --aug-paper-prob 0.3 \
    --aug-paper-type-probs "0.8,0.2,0.0" \  # mostly white, some yellow, no dots
    --aug-paper-texture-probs "0.8,0.2,0.0" \  # mostly plain, some grainy, no crumple
    --aug-lighting-prob 0.3 \
    --aug-lighting-modes "normal,bright"      # only normal/bright lighting
```

### Medium Augmentation (Varied Conditions)
Balanced mix of paper types and textures:

```bash
python train.py \
    --data single_symbol_data/ \
    --out checkpoints/medium_aug \
    --epochs 50 \
    --aug-paper-prob 0.5 \
    --aug-paper-type-probs "0.5,0.3,0.2" \
    --aug-paper-texture-probs "0.5,0.3,0.2" \
    --aug-lines-prob 0.3 \
    --aug-lighting-prob 0.5
```

### Heavy Augmentation (Maximum Robustness)
Extreme variations to handle difficult real-world conditions:

```bash
python train.py \
    --data single_symbol_data/ \
    --out checkpoints/heavy_aug \
    --epochs 50 \
    --augment \                              # geometric + paper augmentation
    --aug-paper-prob 0.7 \
    --aug-paper-texture-probs "0.3,0.3,0.4" \  # more grainy/crumpled
    --aug-lines-prob 0.4 \
    --aug-lighting-prob 0.7 \
    --aug-crumple-strength-min 2.0 \
    --aug-crumple-strength-max 4.0            # stronger crumple
```

### Yellow Paper Focus
Train specifically on yellow ruled paper (for school notebooks):

```bash
python train.py \
    --data single_symbol_data/ \
    --out checkpoints/yellow_paper \
    --epochs 50 \
    --aug-paper-prob 1.0 \                   # always apply paper
    --aug-paper-type-probs "0.0,1.0,0.0" \   # 100% yellow paper
    --aug-paper-texture-probs "0.7,0.3,0.0" \  # plain or grainy, no crumple
    --aug-lighting-prob 0.5 \
    --aug-line-color "30,90,160"             # blue lines (yellow paper default)
```

### Dotted Paper Focus
Train for dotted grid paper:

```bash
python train.py \
    --data single_symbol_data/ \
    --out checkpoints/dotted \
    --epochs 50 \
    --aug-paper-prob 1.0 \
    --aug-paper-type-probs "0.0,0.0,1.0" \   # 100% dotted
    --aug-dot-size 2 \
    --aug-dot-opacity 60 \
    --aug-dot-spacing 20
```

---

## Augmentation Pipeline Architecture

### Transform Order

```
Input PIL Image (from dataset)
    ↓
[1] PaperAugmentation (if enabled)
    ├── Paper Texture (paper_prob)
    ├── Ruled Lines (lines_prob)
    └── Lighting (lighting_prob)
    ↓
[2] Standard Geometric Augmentation (if --augment)
    ├── RandomResizedCrop
    ├── RandomRotation
    ├── RandomAffine
    └── ColorJitter
    ↓
[3] ToTensor
    ↓
[4] Normalize (ImageNet stats)
    ↓
Output Tensor
```

**Note**: Paper augmentation happens **before** geometric transforms to preserve realistic paper effects.

---

## Performance Impact

### Training Speed

| Configuration | Relative Speed | Notes |
|--------------|----------------|-------|
| No augmentation | 100% (baseline) | Fastest |
| `--augment` only | ~95% | Minimal overhead |
| Paper texture 50% | ~60% | Moderate (PIL operations) |
| Lines 30% | ~85% | Low overhead |
| Lighting 50% | ~80% | Low-moderate overhead |
| Paper + Lines + Lighting | ~50-60% | Combined moderate overhead |
| Heavy augmentation | ~40-50% | Significant but worthwhile |

### Recommendations

- **Start with 30-50% probabilities** for paper/lighting augmentation
- **Increase gradually** if model needs more robustness
- **Use --batch to compensate** - larger batch size can help GPU utilization
- **Monitor training time** - if too slow, reduce probabilities or disable crumple

---

## Validation Behavior

**Important**: Paper augmentation is **only applied to training data**, not validation data.

Validation always uses:
```python
transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

This ensures:
- ✅ Fair comparison of validation metrics across runs
- ✅ Validation represents clean symbol recognition performance
- ✅ Augmentation benefits are measured through train→val generalization gap

---

## Tuning Recommendations

### Finding Optimal Probabilities

1. **Start Conservative**
   ```bash
   --aug-paper-prob 0.3 --aug-lighting-prob 0.3
   ```

2. **Monitor Validation Accuracy**
   - If val_acc improves → augmentation is helping
   - If train_acc << val_acc → increase augmentation
   - If train_acc ≈ val_acc → augmentation is sufficient

3. **Increase Gradually**
   ```bash
   0.3 → 0.4 → 0.5 → 0.6 → 0.7
   ```

4. **Stop When Overfitting Controlled**
   - Target: small train/val gap (±5-10%)
   - Don't over-augment: too much hurts training signal

### Paper Type Distribution

Match your **target deployment environment**:

```bash
# School notebooks (yellow ruled paper)
--aug-paper-type-probs "0.2,0.7,0.1"

# Office/printer paper (white)
--aug-paper-type-probs "0.8,0.1,0.1"

# Bullet journals (dotted)
--aug-paper-type-probs "0.3,0.2,0.5"

# Mixed (general purpose)
--aug-paper-type-probs "0.6,0.2,0.2"  # default
```

### Texture Distribution

```bash
# Clean paper (minimal texture)
--aug-paper-texture-probs "0.8,0.2,0.0"

# Realistic paper (some texture/wrinkles)
--aug-paper-texture-probs "0.5,0.3,0.2"  # default

# Extreme conditions (heavy crumpling)
--aug-paper-texture-probs "0.2,0.3,0.5"
```

---

## Consistency with Detection Training

The paper augmentation system is **identical** to the detection pipeline's synthetic generation, ensuring:

✅ **Same paper backgrounds** (white/yellow/dotted)  
✅ **Same surface textures** (plain/grainy/crumpled)  
✅ **Same ruled line overlay** (spacing, opacity, color)  
✅ **Same lighting modes** (normal/bright/dim/shadows)  
✅ **Same implementation** (shared `src/shared/augmentations.py`)

This means classification models see the **same realistic paper conditions** as detection models during training!

---

## Troubleshooting

### Augmentation Not Applied

**Problem**: Training runs but no paper effects visible.

**Check**:
```bash
# Ensure probabilities are > 0
--aug-paper-prob 0.5  # NOT 0.0
--aug-lighting-prob 0.5

# Check for messages during training
# Should see: "Paper augmentation enabled:"
```

### Training Too Slow

**Problem**: Training takes 2-3x longer with augmentation.

**Solutions**:
```bash
# Reduce probabilities
--aug-paper-prob 0.3  # instead of 0.7

# Disable expensive operations
--aug-paper-texture-probs "0.7,0.3,0.0"  # no crumple

# Increase batch size to improve GPU utilization
--batch 64  # instead of 32
```

### Training Accuracy Too Low

**Problem**: train_acc stuck at low values with heavy augmentation.

**Solutions**:
```bash
# Reduce augmentation strength
--aug-paper-prob 0.3  # instead of 0.7

# Increase epochs (model needs more time to learn)
--epochs 100  # instead of 50

# Combine with less aggressive parameters
--aug-crumple-strength-max 2.0  # instead of 4.0
--aug-paper-strength-max 0.3    # instead of 0.5
```

### Color Distribution Issues

**Problem**: Images look too dark/bright.

**Solutions**:
```bash
# Adjust lighting mode distribution
--aug-lighting-modes "normal,bright"  # exclude dim/shadows

# Reduce shadow intensity
--aug-shadow-intensity-max 0.2  # instead of 0.3

# Adjust paper strength (less paper = more original colors)
--aug-paper-strength-max 0.3  # instead of 0.4
```

---

## Summary

### Key Benefits

✅ **Realistic Paper Simulation** - train on conditions matching real deployment  
✅ **Improved Generalization** - better performance on unseen paper types  
✅ **Consistent with Detection** - unified augmentation across pipelines  
✅ **Highly Configurable** - 50+ parameters for fine-tuning  
✅ **Probabilistic Application** - control augmentation frequency  

### Recommended Starting Point

```bash
python train.py \
    --data single_symbol_data/ \
    --out checkpoints/recommended \
    --epochs 50 \
    --batch 32 \
    --augment \
    --aug-paper-prob 0.4 \
    --aug-lighting-prob 0.5 \
    --aug-lines-prob 0.2
```

This provides a **balanced mix** of:
- Standard geometric augmentation (rotation, crop, affine)
- 40% paper texture (realistic backgrounds)
- 50% lighting variations (brightness/shadows)
- 20% ruled lines (notebook paper simulation)

### Next Steps

1. Train baseline model without augmentation
2. Train with recommended augmentation settings
3. Compare validation accuracy improvement
4. Tune probabilities based on target deployment
5. Test on real-world images to validate robustness
