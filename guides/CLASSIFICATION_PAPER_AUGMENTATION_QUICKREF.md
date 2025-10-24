# Classification Paper Augmentation - Quick Reference

## Most Common Use Cases

### 1. General Purpose Training
```bash
python train.py \
    --data single_symbol_data/ \
    --out checkpoints/general \
    --epochs 50 \
    --augment \
    --aug-paper-prob 0.4 \
    --aug-lighting-prob 0.5
```

### 2. Yellow Ruled Paper (School Notebooks)
```bash
python train.py \
    --data single_symbol_data/ \
    --out checkpoints/yellow \
    --epochs 50 \
    --aug-paper-prob 0.8 \
    --aug-paper-type-probs "0.0,1.0,0.0" \
    --aug-line-color "30,90,160"
```

### 3. Clean White Paper
```bash
python train.py \
    --data single_symbol_data/ \
    --out checkpoints/white \
    --epochs 50 \
    --aug-paper-prob 0.5 \
    --aug-paper-type-probs "1.0,0.0,0.0" \
    --aug-paper-texture-probs "0.8,0.2,0.0"
```

### 4. Heavy Augmentation (Maximum Robustness)
```bash
python train.py \
    --data single_symbol_data/ \
    --out checkpoints/heavy \
    --epochs 50 \
    --augment \
    --aug-paper-prob 0.7 \
    --aug-lines-prob 0.4 \
    --aug-lighting-prob 0.7
```

## All Arguments

### Paper Texture (--aug-paper-*)
| Argument                          | Default         | Range  | Description                                             |
| --------------------------------- | --------------- | ------ | ------------------------------------------------------- |
| `--aug-paper-prob`                | `0.0`           | `0..1` | Probability to apply paper texture                      |
| `--aug-paper-type-probs`          | `"0.6,0.2,0.2"` | CSV    | Probabilities for `[white, yellow, dotted]` paper types |
| `--aug-paper-texture-probs`       | `"0.5,0.3,0.2"` | CSV    | Probabilities for `[plain, grainy, crumpled]` textures  |
| `--aug-paper-strength-min`        | `0.2`           | `0..1` | Minimum paper blend strength                            |
| `--aug-paper-strength-max`        | `0.4`           | `0..1` | Maximum paper blend strength                            |
| `--aug-paper-yellow-strength-min` | `0.3`           | `0..1` | Minimum yellow paper strength                           |
| `--aug-paper-yellow-strength-max` | `0.5`           | `0..1` | Maximum yellow paper strength                           |
| `--aug-crumple-strength-min`      | `1.0`           | `0..5` | Minimum crumple warp strength                           |
| `--aug-crumple-strength-max`      | `3.0`           | `0..5` | Maximum crumple warp strength                           |
| `--aug-crumple-mesh-overlap`      | `2`             | `1..4` | Mesh tile overlap (pixels)                              |

### Paper Lines (--aug-line-*)
| Argument                   | Default   | Range     | Description                        |
| -------------------------- | --------- | --------- | ---------------------------------- |
| `--aug-lines-prob`         | `0.0`     | `0..1`    | Probability to overlay ruled lines |
| `--aug-line-spacing-min`   | `24`      | `10..100` | Minimum spacing between lines      |
| `--aug-line-spacing-max`   | `32`      | `10..100` | Maximum spacing between lines      |
| `--aug-line-opacity-min`   | `30`      | `0..255`  | Minimum line opacity (alpha)       |
| `--aug-line-opacity-max`   | `60`      | `0..255`  | Maximum line opacity (alpha)       |
| `--aug-line-thickness-min` | `1`       | `1..5`    | Minimum line thickness (pixels)    |
| `--aug-line-thickness-max` | `2`       | `1..5`    | Maximum line thickness (pixels)    |
| `--aug-line-jitter-min`    | `1`       | `0..10`   | Minimum vertical jitter (pixels)   |
| `--aug-line-jitter-max`    | `3`       | `0..10`   | Maximum vertical jitter (pixels)   |
| `--aug-line-color`         | `"0,0,0"` | RGB       | Line color as `R,G,B`              |

### Lighting (--aug-lighting-*, --aug-shadow-*)
| Argument                     | Default                       | Range    | Description                                 |
| ---------------------------- | ----------------------------- | -------- | ------------------------------------------- |
| `--aug-lighting-prob`        | `0.0`                         | `0..1`   | Probability to apply lighting augmentations |
| `--aug-lighting-modes`       | `"normal,bright,dim,shadows"` | CSV      | Lighting modes to sample from               |
| `--aug-brightness-jitter`    | `0.03`                        | `0..0.2` | Brightness jitter strength                  |
| `--aug-contrast-jitter`      | `0.03`                        | `0..0.2` | Contrast jitter strength                    |
| `--aug-shadow-intensity-min` | `0.0`                         | `0..1`   | Minimum shadow intensity                    |
| `--aug-shadow-intensity-max` | `0.3`                         | `0..1`   | Maximum shadow intensity                    |

| Argument            | Default | Range    | Description                   |
| ------------------- | ------- | -------- | ----------------------------- |
| `--aug-dot-size`    | `1`     | `1..5`   | Dot radius in pixels          |
| `--aug-dot-opacity` | `50`    | `0..255` | Dot alpha (transparency)      |
| `--aug-dot-spacing` | `18`    | `10..50` | Spacing between dots (pixels) |

## Tips

### Performance
- Paper texture: ~40% slower
- Lines: ~15% slower  
- Lighting: ~20% slower
- Combined: ~50% slower

**Solution**: Start with prob=0.3-0.5, increase batch size

### Probability Tuning
- **Conservative**: 0.3 probabilities
- **Balanced**: 0.5 probabilities
- **Aggressive**: 0.7 probabilities

### Common Combinations
```bash
# Light (fast training)
--aug-paper-prob 0.3 --aug-lighting-prob 0.3

# Medium (balanced)
--aug-paper-prob 0.5 --aug-lines-prob 0.3 --aug-lighting-prob 0.5

# Heavy (slow but robust)
--aug-paper-prob 0.7 --aug-lines-prob 0.5 --aug-lighting-prob 0.7
```
