# sulat-baybayin

Sulat Baybayin is an AI-powered toolkit for generating, training, and running Baybayin OCR/detection and classification models. This repository contains utilities to synthesize sentence images, train detection models (two-stage workflows supported), run inference that emits annotated images and assembled sentences, and small helpers for dataset preparation.

## Quickstart

- Install dependencies:

```cmd
python -m pip install -r requirements.txt
```

- Generate synthetic sentence images (see `generate_synthetic_sentences.py` for detailed flags):

```cmd
python generate_synthetic_sentences.py --help

python generate_synthetic_sentences.py --out sentences_data_synth --n 1000 --paper-lines-prob 0.5
```

## Detection training

The repository provides `train_detection.py` which delegates to `src.detection.train`. It supports a two-stage training flow (head-only then finetune) and a few checkpointing behaviors you should be aware of.

Typical two-stage workflow examples:

Stage 1 (train head only):

```cmd
python train_detection.py --data sentences_data_synth --ann annotations/synthetic_annotations_noheader.csv \
	--epochs 10 --batch 2 --num-workers 2 --pin-memory --amp --freeze-backbone \
	--lr 0.001 --lr-head 0.005 --momentum 0.9 --weight-decay 0.0005 \
	--no-batch-eval --early-stop-monitor val_loss --early-stop-min-delta 0.0005 --early-stop-patience 3 \
	--save-last --out checkpoints/detection/stage1
```

Stage 2 (unfreeze backbone, finetune):

```cmd
python train_detection.py --data sentences_data_synth --ann annotations/synthetic_annotations_noheader.csv \
	--resume checkpoints/detection/stage1/best.pth --epochs 20 --batch 2 --num-workers 2 --pin-memory --amp \
	--lr 2e-4 --lr-backbone 5e-5 --lr-head 1e-4 --momentum 0.9 --weight-decay 0.0005 \
	--no-batch-eval --early-stop-monitor val_loss --early-stop-min-delta 0.00025 --early-stop-patience 6 \
	--save-last --out checkpoints/detection/stage2
```

Notes on checkpointing and early stopping
- `--val-ann`: validation annotations file. Validation loss is only computed when you provide `--val-ann`. If you specify `--early-stop-monitor val_loss` without `--val-ann`, the training script will emit a warning and val_loss will be treated as unavailable (early-stopping on val_loss will be skipped).
- `--save-last`: when provided the script writes `last.pth` at the end of every epoch (overwriting the previous `last.pth`). This is useful for resuming or debugging.
- `best_seen.pth`: the script saves `best_seen.pth` whenever the monitored metric improves during training (this preserves the best-observed model by the chosen monitor).
- `early_stop.pth`: when early stopping triggers the script saves `early_stop.pth` (and `final_epoch_{N}.pth`).
- `best.pth`: at the end of training (normal completion or early stop) the script writes `best.pth` containing the final model state (the model at training stop). Note: this repository also preserves `best_seen.pth` to track the best observed model during training.

Resuming training
- Use `--resume path/to/checkpoint.pth` to load a model checkpoint. If you also want to restore optimizer state use `--resume_optimizer` (optional). Be careful: if you change parameter groups (for example by unfreezing the backbone and using different LR groups) you should avoid restoring the optimizer state because optimizer parameter-group mismatches can cause unexpected behavior.
- The script sets `start_epoch = ckpt['epoch'] + 1` when `epoch` is present in the checkpoint. If you want to run N more epochs after a saved checkpoint make sure to set `--epochs` appropriately (e.g., `--epochs ckpt_epoch + N`), or inspect the checkpoint epoch with:

```cmd
python -c "import torch; ck=torch.load(r'checkpoints/detection/colab_run8/stage1/best.pth', map_location='cpu'); print(ck.get('epoch'))"
```

Early-stop monitors
- `--early-stop-monitor` supports `val_loss`, `train_loss`, and `acc`. `val_loss` requires `--val-ann` to be meaningful. `train_loss` and `acc` are available without a validation split.

Creating validation splits
- If you don't yet have a validation CSV you can create one using the helper `tools/split_annotations.py` which splits by image (keeps all boxes from an image together) and writes train/val files.

## Inference (detection)

Use `infer_detection.py` for detection inference. It will write annotated images, per-image text results and optional compiled outputs.

Example:

```cmd
python infer_detection.py --ckpt checkpoints\detection\stage2\best.pth --input sentences_testing\ --out detections\test1 --thresh 0.5 --compile-inferred
```

Key outputs produced by inference (when `--compile-inferred` is used):
- Annotated PNGs in the output folder (with boxes and labels)
- Per-image text files listing detected boxes and candidate text
- `compiled_inferred.txt` — assembled sentence predictions
- `compiled_annotations.csv` — flattened CSV of image_path,x1,y1,x2,y2,label,confidence_score

## Synthetic data generator highlights

`generate_synthetic_sentences.py` supports symbol normalization, on-disk caching of normalized glyphs, optional erosion behavior (for shadows and glyphs), and a ruled-paper overlay to make synthetic data look more like photographed text.

### CLI Flags
Below are the most common flags exposed by `generate_synthetic_sentences.py`. Run `python generate_synthetic_sentences.py --help` for the full list.

- `--count` (int, default 500)
	- Number of synthetic sentence images to generate.
- `--out-dir` (str, default `sentences_data_synth`)
	- Output root directory; images are written to `<out-dir>/images` and annotations to `<out-dir>/annotations.csv` (unless `--ann` is provided).
- `--ann` (path, default None)
	- Path to write the annotations CSV. If omitted, the generator writes `<out-dir>/annotations.csv`.
- `--min_symbols`, `--max_symbols` (int, defaults 3 and 8)
	- Min/max number of symbols (glyphs) per generated sentence image.

Normalization options
- `--symbol-height-frac` (float, default 0.55)
	- Fraction of the canvas height used as the target symbol height when normalizing glyphs.
- `--use-cache` (flag)
	- Cache normalized symbols to disk (saves time on repeated runs). Cached entries are stored in `--cache-dir`.
- `--cache-dir` (str, default `.symbol_cache`)
	- Directory used to store cached normalized glyph PNGs.
- `--bg-thresh-pct` (float, default 99.0)
	- Percentile used to determine the background luminance cutoff when deriving masks from raster glyphs.

Erosion (shadow and glyph) options
- `--erode-shadow` (flag)
	- If set, the script may erode the glyph mask before creating the shadow to produce a thinner shadow. Controlled by a thickness heuristic and probability flags.
- `--erode-shadow-min-thickness` (float, default 2.5)
	- Minimum estimated stroke thickness required to permit shadow erosion.
- `--erode-glyph` (flag)
	- If set, the script may apply destructive erosion to the glyph alpha mask (used sparingly to simulate ink wear).
- `--erode-glyph-min-thickness` (float, default 4.0)
	- Minimum estimated stroke thickness required to permit glyph erosion.
- `--erode-shadow-prob`, `--erode-glyph-prob` (float, default 1.0)
	- Probabilities (0..1) controlling whether erosion is applied when thresholds are met.

Paper-line styling (ruled-paper overlay)
- `--paper-lines-prob` (float, default 0.0)
	- Per-image probability (0..1) of overlaying ruled-paper lines. When >0, a random subset of generated images will receive a paper-line overlay.
- `--line-spacing` (int, default 28)
	- Pixel spacing between horizontal ruled lines.
- `--line-opacity` (int, default 40)
	- Alpha opacity for the ruled lines (0-255).
- `--line-thickness` (int, default 1)
	- Line thickness in pixels.
- `--line-jitter` (int, default 2)
	- Per-line vertical jitter in pixels to avoid perfectly uniform lines.
- `--line-color` (str, default `0,0,0`)
	- RGB color for lines as comma-separated integers (e.g., `0,0,0`).

Other notes & tips
- Caching: enable `--use-cache` when you will regenerate datasets repeatedly from the same symbol images — it can substantially reduce preprocessing time.
- Validation / annotation: the generator writes an annotations CSV with rows `[image_path,x1,y1,x2,y2,label]`. This format is compatible with `train_detection.py`'s CSV reader (no header required for the trainer, but the generator writes a header by default).
- Reproducibility: the generator uses Python's `random` module; set the environment or call site seed externally if you need deterministic image sets.
- Examples:

```cmd
# generate 1000 images with paper lines on ~50% of images
python generate_synthetic_sentences.py --count 1000 --out-dir sentences_data_synth --paper-lines-prob 0.5 --use-cache

# produce a small validation set and write annotations to a custom path
python generate_synthetic_sentences.py --count 200 --out-dir sentences_data_synth_val --ann annotations/synth_val.csv
```

If you'd like I can add a short preview utility that generates N images and shows them in a simple HTML gallery for quick inspection.

## Helpers
- `tools/split_annotations.py` — split an annotations CSV into train/val/test by image
- `tools/write_noheader_annotations.py` — convert CSVs into the trainer's no-header format

## Tips
- If you rely on `val_loss` for early stopping, keep a reasonably sized validation set. Detection val_loss can be noisy; consider monitoring a detection metric (AP) when possible.
- Use `--save-last` when you want an automatic rolling checkpoint with minimal effort.
- Avoid restoring optimizer state when you change parameter groups (e.g., unfreeze backbone) — instead start optimizer fresh for stage2.

If you'd like, I can expand this README with a full examples section, a quickstart notebook, or a short troubleshooting guide for common checkpoint/resume issues.
