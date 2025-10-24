# YAML Configuration System - Complete Implementation

## 🎉 Implementation Complete!

The YAML configuration system is now fully integrated across all training and data generation scripts. This system allows you to manage complex parameter sets in version-controlled YAML files instead of long command-line arguments.

---

## ✅ What Was Implemented

### 1. Core Utilities (`src/shared/config_manager.py`)
- ✅ `generate_yaml_template()` - Generate YAML config from defaults
- ✅ `load_yaml_config()` - Load and parse YAML files
- ✅ `merge_configs()` - Merge YAML with CLI arguments (CLI takes priority)
- ✅ `wait_for_user_edit()` - Interactive workflow for editing configs
- ✅ `flatten_dict()` / `unflatten_dict()` - Handle nested config structures
- ✅ `validate_param_range()` - Parameter validation utilities

### 2. Script Integration
- ✅ `generate_synthetic_sentences.py` - Synthetic data generation
- ✅ `src/classification/train.py` - Classification training
- ✅ `src/detection/train.py` - Detection training

All scripts now support:
- `--args-input <path>` - Load config from YAML file
- `--no-wait` - Skip interactive edit prompt
- `--regen-args` - Force regenerate YAML template

### 3. Example Configurations

**Synthetic Data Generation:**
- `configs/synth_light.yaml` - Minimal augmentation (1K images)
- `configs/synth_heavy.yaml` - Aggressive augmentation (10K images)

**Classification Training:**
- `configs/classification_baseline.yaml` - Standard training without paper aug
- `configs/classification_paper_aug.yaml` - With paper texture/lines/lighting

**Detection Training:**
- `configs/detection_baseline_stage1.yaml` - Stage 1 (frozen backbone)
- `configs/detection_baseline_stage2.yaml` - Stage 2 (full training)
- `configs/detection_baseline_stage3.yaml` - Stage 3 (fine-tuning)

### 4. Documentation
- ✅ `configs/README.md` - Comprehensive usage guide
- ✅ `test_yaml_config.py` - Test suite (6/6 tests passing)
- ✅ All example configs validated and tested

### 5. Dependencies
- ✅ Added `PyYAML>=6.0` to `requirements.txt`

---

## 🚀 Quick Start

### Generate Synthetic Data with YAML

**Option 1: Use existing config**
```bash
python generate_synthetic_sentences.py --args-input configs/synth_heavy.yaml
```

**Option 2: Generate new config template**
```bash
python generate_synthetic_sentences.py --args-input configs/my_config.yaml
# Script will generate template, pause for you to edit, then continue
```

**Option 3: Generate template without waiting**
```bash
python generate_synthetic_sentences.py \
    --args-input configs/my_config.yaml \
    --no-wait
```

### Train Classification Model

**Baseline training:**
```bash
python -m src.classification.train \
    --data single_symbol_data \
    --args-input configs/classification_baseline.yaml
```

**With paper augmentation:**
```bash
python -m src.classification.train \
    --data single_symbol_data \
    --args-input configs/classification_paper_aug.yaml
```

### Train Detection Model (3 Stages)

**Stage 1:**
```bash
python train_detection.py \
    --data sentences_data_synth \
    --ann annotations/synthetic_annotations.csv \
    --args-input configs/detection_baseline_stage1.yaml
```

**Stage 2:**
```bash
python train_detection.py \
    --data sentences_data_synth \
    --ann annotations/synthetic_annotations.csv \
    --args-input configs/detection_baseline_stage2.yaml \
    --resume checkpoints/detection/baseline/stage1/best.pth
```

**Stage 3:**
```bash
python train_detection.py \
    --data sentences_data_synth \
    --ann annotations/synthetic_annotations.csv \
    --args-input configs/detection_baseline_stage3.yaml \
    --resume checkpoints/detection/baseline/stage2/best.pth
```

---

## 💡 Key Features

### 1. CLI Overrides YAML
CLI arguments always take precedence over YAML values:

```bash
# Use YAML config but override specific values
python generate_synthetic_sentences.py \
    --args-input configs/synth_heavy.yaml \
    --count 5000 \
    --out-dir custom_output
```

### 2. Interactive Workflow
When YAML file doesn't exist, script generates template and pauses:

```bash
$ python generate_synthetic_sentences.py --args-input configs/new.yaml

YAML config file not found at configs/new.yaml
Generating template with current defaults...
✓ Generated template: configs/new.yaml

Please edit the YAML file to configure your parameters.
Press Enter when ready to continue...
```

### 3. Non-Interactive Mode
Skip the pause for automation/scripts:

```bash
python generate_synthetic_sentences.py \
    --args-input configs/new.yaml \
    --no-wait
```

### 4. Force Regeneration
Update existing config with new defaults:

```bash
python generate_synthetic_sentences.py \
    --args-input configs/existing.yaml \
    --regen-args
```

---

## 📊 Test Results

All tests passing ✅:

```
TEST 1: Flatten/Unflatten Dict          ✓ Passed
TEST 2: Generate YAML Template          ✓ Passed
TEST 3: Load YAML and Merge with CLI    ✓ Passed
TEST 4: Load Example Configs            ✓ Passed
TEST 5: Full Workflow Simulation        ✓ Passed
TEST 6: Edge Cases                      ✓ Passed

Total: 6/6 tests passed 🎉
```

Run tests:
```bash
python test_yaml_config.py
```

---

## 📁 File Structure

```
sulat-baybayin/
├── configs/
│   ├── README.md                           # Usage guide
│   ├── synth_light.yaml                    # Light synthetic data config
│   ├── synth_heavy.yaml                    # Heavy synthetic data config
│   ├── classification_baseline.yaml        # Classification baseline
│   ├── classification_paper_aug.yaml       # Classification with paper aug
│   ├── detection_baseline_stage1.yaml      # Detection stage 1
│   ├── detection_baseline_stage2.yaml      # Detection stage 2
│   └── detection_baseline_stage3.yaml      # Detection stage 3
├── src/
│   └── shared/
│       └── config_manager.py               # YAML utilities
├── generate_synthetic_sentences.py         # Integrated YAML support
├── src/classification/train.py             # Integrated YAML support
├── src/detection/train.py                  # Integrated YAML support
├── test_yaml_config.py                     # Test suite
└── requirements.txt                        # Added PyYAML>=6.0
```

---

## 🔧 How It Works

### 1. User runs script with `--args-input`

```bash
python script.py --args-input configs/my_config.yaml
```

### 2. Script checks if YAML exists

**If exists:**
- Load YAML config
- Merge with CLI args (CLI overrides YAML)
- Continue execution

**If doesn't exist:**
- Generate template with defaults
- Wait for user to edit (unless `--no-wait`)
- Load edited config
- Merge with CLI args
- Continue execution

### 3. Merging Logic

```python
# YAML config
yaml_config = {
    'count': 1000,
    'out_dir': 'yaml_output',
    'paper_type': 'white'
}

# CLI args (from argparse)
cli_args = {
    'count': 2000,      # Override YAML
    'out_dir': 'yaml_output',  # Same as YAML
    'new_arg': 'value'  # New from CLI
}

# Merged result
merged = {
    'count': 2000,           # CLI override
    'out_dir': 'yaml_output', # From YAML
    'paper_type': 'white',   # From YAML
    'new_arg': 'value'       # From CLI
}
```

---

## 🎯 Benefits

### Before (Long CLI commands):
```bash
python generate_synthetic_sentences.py \
    --count 10000 \
    --out-dir sentences_data_synth_run11 \
    --paper-type-mix "0.5,0.3,0.2" \
    --paper-texture crumpled \
    --paper-strength 0.35 \
    --crumple-strength 2.5 \
    --paper-lines-prob 0.4 \
    --line-opacity 45 \
    --lighting shadows \
    --shadow-intensity 0.25 \
    --erode-shadow \
    --erode-shadow-prob 0.7 \
    --erode-glyph \
    --erode-glyph-prob 0.5 \
    --ink-darken-min 0.75 \
    --ink-darken-max 0.96 \
    --thin-alpha-gain 1.3 \
    --thin-alpha-floor 140
    # ... 40+ more parameters!
```

### After (YAML config):
```bash
# Short, readable command
python generate_synthetic_sentences.py \
    --args-input configs/synth_heavy.yaml

# Or override specific values
python generate_synthetic_sentences.py \
    --args-input configs/synth_heavy.yaml \
    --count 20000 \
    --out-dir custom_output
```

### Advantages:
✅ **Reproducible** - Exact configs in version control  
✅ **Shareable** - Send config file instead of long commands  
✅ **Readable** - YAML is human-friendly  
✅ **Maintainable** - Edit config file instead of command history  
✅ **Flexible** - CLI can still override any value  
✅ **Documented** - Configs serve as documentation  

---

## 🔄 Migration from Old Commands

### Old Style:
```bash
python generate_synthetic_sentences.py \
    --count 10000 \
    --out-dir sentences_data_synth_run11 \
    --paper-type-mix "0.5,0.3,0.2" \
    --paper-texture crumpled \
    # ... many more flags
```

### New Style (Option 1: Pure YAML):
```bash
# 1. Generate template
python generate_synthetic_sentences.py \
    --args-input configs/run11.yaml \
    --regen-args

# 2. Edit configs/run11.yaml to set all parameters

# 3. Run with config
python generate_synthetic_sentences.py \
    --args-input configs/run11.yaml
```

### New Style (Option 2: Hybrid):
```bash
# Use base config but override changing values
python generate_synthetic_sentences.py \
    --args-input configs/synth_heavy.yaml \
    --count 10000 \
    --out-dir sentences_data_synth_run11
```

---

## 📝 Example YAML Config

```yaml
# Synthetic data generation config
# configs/synth_heavy.yaml

# Output & Generation
count: 10000
out_dir: sentences_data_synth_heavy
ann: null
append: false
min_symbols: 3
max_symbols: 10

# Paper Type & Texture
paper_type: white
paper_type_mix: "0.5,0.3,0.2"  # white, yellow, dotted
paper_texture: crumpled
paper_strength: 0.35
crumple_strength: 2.5

# Ruled Lines
paper_lines_prob: 0.4
line_spacing: 28
line_opacity: 45
line_thickness: 1

# Lighting
lighting: shadows
brightness_jitter: 0.04
contrast_jitter: 0.04
shadow_intensity: 0.25

# Ink Appearance
ink_color: black
ink_darken_min: 0.75
ink_darken_max: 0.96
thin_alpha_gain: 1.3
thin_alpha_floor: 140

# ... more parameters
```

---

## 🐛 Troubleshooting

### Issue: "YAML file not found"
**Solution**: Script will generate template automatically. Review and press Enter.

### Issue: "Changes to YAML not reflected"
**Solution**: Make sure CLI isn't overriding the value. CLI always wins.

### Issue: "Syntax error in YAML"
**Solution**: Check YAML syntax (indentation matters!). Use YAML validator or linter.

### Issue: "Parameter not recognized"
**Solution**: Ensure parameter name matches expected format (underscores, not hyphens).

---

## 🎓 Best Practices

### 1. Version Control Your Configs
```bash
git add configs/experiment_2025-01-15.yaml
git commit -m "Add config for experiment X"
```

### 2. Name Configs Descriptively
- ✅ `configs/run11_heavy_aug.yaml`
- ✅ `configs/classification_paper_aug_v2.yaml`
- ❌ `configs/config1.yaml`
- ❌ `configs/test.yaml`

### 3. Add Comments to Configs
```yaml
# Using higher crumple for handwriting simulation
crumple_strength: 4.5

# Disable erosion to preserve thin strokes
erode_glyph: false
```

### 4. Use Base Configs + CLI Overrides
```bash
# Keep stable params in YAML, override changing ones
python script.py \
    --args-input configs/base.yaml \
    --out checkpoints/experiment_01 \
    --lr 0.001
```

### 5. Test Configs Before Long Runs
```bash
# Quick test with reduced count
python generate_synthetic_sentences.py \
    --args-input configs/synth_heavy.yaml \
    --count 100 \
    --out-dir test_output
```

---

## 📚 Related Documentation

- `configs/README.md` - Detailed usage guide for configs
- `GDRIVE_AUTO_BACKUP_GUIDE.md` - Google Drive backup for training
- `CLASSIFICATION_PAPER_AUGMENTATION.md` - Paper augmentation details

---

## Summary

✅ **YAML config system fully implemented and tested**  
✅ **All scripts support `--args-input` flag**  
✅ **7 example configs provided**  
✅ **Comprehensive documentation created**  
✅ **All tests passing (6/6)**  

### Quick Commands:

```bash
# Generate synthetic data
python generate_synthetic_sentences.py --args-input configs/synth_heavy.yaml

# Train classification
python -m src.classification.train --data single_symbol_data --args-input configs/classification_paper_aug.yaml

# Train detection
python train_detection.py --data sentences_data_synth --ann annotations/synthetic_annotations.csv --args-input configs/detection_baseline_stage1.yaml

# Run tests
python test_yaml_config.py
```

**Ready to use!** 🚀
