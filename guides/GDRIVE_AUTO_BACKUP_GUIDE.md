# Google Drive Auto-Backup for Training

## 🔄 How It Works

The training script now **automatically backs up checkpoints to Google Drive after every epoch** when you use the `--save-last` and `--gdrive-backup` flags together.

### Backup Frequency

✅ **After every epoch** (when using `--save-last`)  
✅ **At the end of training** (final best.pth)  
✅ **On early stopping** (when training stops early)  

This means if your Colab session crashes, you can **resume from the last completed epoch** using the backup in Google Drive!

---

## 🚀 Usage

### Basic Command Structure
```bash
python train_detection.py \
    [training args...] \
    --save-last \
    --gdrive-backup /content/drive/MyDrive/SulatBaybayin/
```

### Run 11 Commands (with per-epoch backup)

All commands are the same as before - the per-epoch backup is automatic when you use `--save-last` + `--gdrive-backup`!

#### Stage 1
```bash
python train_detection.py --data sentences_data_synth_run11 --ann annotations/synthetic_annotations_run11_noheader.csv --epochs 15 --batch 4 --num-workers 4 --pin-memory --amp --freeze-backbone --lr 0.002 --lr-head 0.008 --momentum 0.95 --weight-decay 0.001 --no-batch-eval --early-stop-monitor train_loss --early-stop-min-delta 0.0005 --early-stop-patience 5 --out checkpoints/detection/colab_run11/stage1 --save-last --gdrive-backup /content/drive/MyDrive/SulatBaybayin/
```

#### Stage 2
```bash
python train_detection.py --data sentences_data_synth_run11 --ann annotations/synthetic_annotations_run11_noheader.csv --resume checkpoints/detection/colab_run11/stage1/best.pth --epochs 30 --batch 4 --num-workers 4 --pin-memory --amp --lr 1e-4 --lr-backbone 2e-5 --lr-head 5e-5 --momentum 0.95 --weight-decay 0.001 --no-batch-eval --early-stop-monitor train_loss --early-stop-min-delta 0.0002 --early-stop-patience 8 --out checkpoints/detection/colab_run11/stage2 --save-last --gdrive-backup /content/drive/MyDrive/SulatBaybayin/
```

#### Stage 3
```bash
python train_detection.py --data sentences_data_synth_run11 --ann annotations/synthetic_annotations_run11_noheader.csv --resume checkpoints/detection/colab_run11/stage2/best.pth --epochs 20 --batch 4 --num-workers 4 --pin-memory --amp --lr 3e-5 --lr-backbone 5e-6 --lr-head 1e-5 --momentum 0.95 --weight-decay 0.001 --no-batch-eval --early-stop-monitor train_loss --early-stop-min-delta 0.0001 --early-stop-patience 10 --out checkpoints/detection/colab_run11/stage3 --save-last --gdrive-backup /content/drive/MyDrive/SulatBaybayin/
```

---

## 📁 Backup Structure

### Local (Colab)
```
checkpoints/
└── detection/
    └── colab_run11/
        ├── stage1/
        │   ├── last.pth       (updated every epoch)
        │   └── best.pth       (updated at end)
        ├── stage2/
        │   ├── last.pth
        │   └── best.pth
        └── stage3/
            ├── last.pth
            └── best.pth
```

### Google Drive
```
/content/drive/MyDrive/SulatBaybayin/
└── colab_run11/          (entire run backed up)
    ├── stage1/
    │   ├── last.pth      (synced every epoch)
    │   └── best.pth      (synced at end)
    ├── stage2/
    │   ├── last.pth
    │   └── best.pth
    └── stage3/
        ├── last.pth
        └── best.pth
```

---

## 💡 What Gets Backed Up

### Every Epoch (with --save-last)
- ✅ `last.pth` - current epoch checkpoint (overwritten each epoch)
- ✅ All previous stage folders (complete run history)

### At Training End
- ✅ `best.pth` - final/best checkpoint
- ✅ Full run directory with all stages

---

## 🔄 Resuming After Session Crash

### Scenario: Colab crashes during Stage 2, Epoch 12

#### Option 1: Resume from local (if still available)
```bash
python train_detection.py \
    --resume checkpoints/detection/colab_run11/stage2/last.pth \
    [same args as before...]
```

#### Option 2: Restore from Google Drive
```bash
# Copy back from Drive
!cp -r /content/drive/MyDrive/SulatBaybayin/colab_run11 checkpoints/detection/

# Resume training
python train_detection.py \
    --resume checkpoints/detection/colab_run11/stage2/last.pth \
    --data sentences_data_synth_run11 \
    --ann annotations/synthetic_annotations_run11_noheader.csv \
    --epochs 30 \
    --batch 4 \
    [rest of Stage 2 args...] \
    --save-last \
    --gdrive-backup /content/drive/MyDrive/SulatBaybayin/
```

**Note**: The script will automatically continue from Epoch 13 because `last.pth` stores the epoch number!

---

## 📊 Training Output Example

```
Epoch 1/15: train_loss=2.1234 cls_loss=1.2345 box_loss=0.8889 acc=0.4567 time=205.3s
Saved last.pth (epoch 1)
✓ Backed up to GDrive: /content/drive/MyDrive/SulatBaybayin/colab_run11

Epoch 2/15: train_loss=1.9876 cls_loss=1.1234 box_loss=0.8642 acc=0.5123 time=203.8s
Saved last.pth (epoch 2)
✓ Backed up to GDrive: /content/drive/MyDrive/SulatBaybayin/colab_run11

[... continues every epoch ...]
```

---

## ⚠️ Important Notes

### 1. Requires `--save-last` Flag
The per-epoch backup only happens when you use `--save-last`. This is by design:
- ✅ `--save-last --gdrive-backup` = backup every epoch
- ❌ `--gdrive-backup` only = backup only at end of training

### 2. Network Speed Impact
- Backing up ~500MB checkpoint takes ~10-30 seconds depending on Drive connection
- This happens **after** each epoch completes, so doesn't slow training itself
- Total overhead: ~5-30 seconds per epoch

### 3. Drive Storage
- Each full run backup is ~500MB-2GB depending on model size
- 3-stage run = ~1.5-6GB total
- Make sure you have enough Google Drive space!

### 4. Backup Overwrites
- The backup **overwrites** the previous copy in Drive
- This means you only keep the **latest** state, not a history of every epoch
- If you want to keep multiple checkpoints, use different `--out` paths

---

## 🎯 Best Practices

### ✅ DO
- Always use `--save-last --gdrive-backup` together for maximum safety
- Check Google Drive space before starting long runs
- Test backup with 1-2 epochs before full training
- Keep Drive mounted throughout training (`/content/drive` must exist)

### ❌ DON'T
- Don't manually delete files from Drive during training
- Don't unmount Drive during training
- Don't run multiple training sessions to same `--out` path simultaneously

---

## 🔧 Troubleshooting

### "⚠ GDrive backup failed: ..."
**Cause**: Drive not mounted or path doesn't exist  
**Solution**:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### "Source not found: ..."
**Cause**: Output path format unexpected  
**Solution**: Make sure `--out` follows pattern: `checkpoints/detection/RUN_NAME/stage#`

### Backup Taking Too Long
**Cause**: Large checkpoint files + slow Drive connection  
**Solution**: 
- Reduce model size if possible
- Use smaller batch size (smaller optimizer state)
- Accept the overhead (10-30s per epoch is worth the safety!)

---

## 📈 Performance Impact

| Operation | Time | Impact |
|-----------|------|--------|
| Save checkpoint locally | ~1-2s | Minimal |
| Backup to Google Drive | ~10-30s | Low (happens after epoch) |
| Total overhead per epoch | ~15-40s | ~5-15% |

**For a 200s epoch**: 200s training + 20s backup = 220s total (10% overhead)

**Worth it?** ✅ Absolutely! Losing hours of training due to session timeout costs far more.

---

## Summary

✅ **Automatic backup every epoch** when using `--save-last` + `--gdrive-backup`  
✅ **Resume from any epoch** if session crashes  
✅ **Complete run history** backed up (all stages)  
✅ **Minimal overhead** (~10-30s per epoch)  
✅ **Peace of mind** - never lose training progress again!

Just add `--save-last --gdrive-backup /content/drive/MyDrive/SulatBaybayin/` to your training commands and you're protected! 🛡️
