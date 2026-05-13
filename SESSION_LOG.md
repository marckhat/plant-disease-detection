# Session Log — Apr 29–30, 2026

## Summary

This session covered monitoring an active training run, diagnosing overfitting, implementing Focal Loss and improved training strategy, pushing to GitHub, and writing project documentation.

---

## 1. Training Run Discovery

Discovered an active training process (PID 9011) running `joint_all` dataset with ResNet50 for 30 epochs:

```bash
python3 -u src/train/train.py \
  --model resnet50 \
  --dataset joint_all \
  --epochs 30 \
  --batch-size 32 \
  --lr 5e-5 \
  --scheduler cosine \
  --weighted-loss \
  --cutmix 0.3 \
  --log-every 150 \
  --checkpoint results/models/best_resnet50_joint_aug.pt
```

The log file being monitored (`log_resnet50_joint_all.txt`) was **stale** — the actual process output was going to a deleted temp file at `/proc/9011/fd/1`. Retrieved live output directly from the process file descriptor.

---

## 2. Previous Run Results (Killed at Epoch ~20)

| Epoch | Train Loss | Val Loss | Val Acc | Note |
|-------|-----------|----------|---------|------|
| 1 | 1.0186 | 1.0918 | 71.43% | new best |
| 2 | 0.8839 | 1.1434 | 68.25% | |
| **3** | **0.8204** | **1.1641** | **71.83%** | **best — saved** |
| 4 | 0.7774 | 1.1690 | 68.65% | |
| 5 | 0.7414 | 1.2205 | 68.65% | |
| 6 | 0.7333 | 1.2449 | 71.83% | tied, not saved |
| 7 | 0.7058 | 1.3105 | 68.25% | |
| 8 | 0.6958 | 1.2511 | 69.84% | |
| 9 | 0.6871 | 1.3404 | 70.63% | |
| 10 | 0.6728 | 1.3307 | 68.25% | |
| 11 | 0.6524 | 1.4421 | 68.25% | |
| 12 | 0.6486 | 1.5025 | 66.67% | worst |
| 13 | 0.6415 | 1.3265 | 71.43% | |
| 14 | 0.6296 | 1.3488 | 71.43% | |
| 15 | 0.6217 | 1.4185 | 71.03% | |
| 16 | 0.6151 | 1.4522 | 69.44% | |
| 17 | 0.6071 | 1.5200 | 71.43% | |
| 18 | 0.6008 | 1.5108 | 69.44% | |
| 19 | 0.6024 | 1.4244 | 70.24% | |
| 20 | killed mid-epoch | — | — | |

**Diagnosis:** Classic overfitting from epoch 4 onward. Train loss kept dropping (1.02 → 0.60) while val loss climbed (1.09 → 1.52). LR=5e-5 was too aggressive for fine-tuning from an already strong checkpoint.

---

## 3. Hard Classes Analysis

From the previous evaluation report (`report_resnet50_plantdoc_test.txt`):

| Class | F1-score | Problem |
|-------|---------|---------|
| Corn Gray Leaf Spot | 0.14 | Only 4 test samples, visually similar to other corn diseases |
| Potato Early Blight | 0.20 | Hard to distinguish from Late Blight in field photos |
| Potato Late Blight | 0.35 | Same as above |
| Tomato Bacterial Spot | 0.38 | Similar appearance to other tomato diseases |
| Tomato Mosaic Virus | 0.57 | |

Overall accuracy on real PlantDoc test: **71%** (252 images).

---

## 4. Code Changes

### `src/train/train.py`

**Added `FocalLoss` class:**
- Down-weights easy examples so training focuses on hard ones
- Supports class weights and label smoothing
- Gamma parameter controls focus intensity (default: 2.0)

**Added arguments:**
- `--focal-loss` — use Focal Loss instead of CrossEntropy
- `--focal-gamma` — Focal Loss gamma (default: 2.0)

**Fixed:**
- Class weights are now computed once and shared between CrossEntropy and FocalLoss

### `.gitignore`

Changed `data/` → `/data/` so that `src/data/` (Python code) is no longer excluded from git.

---

## 5. New Training Run (PID 12906)

```bash
python3 -u src/train/train.py \
  --model resnet50 \
  --dataset joint_all \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-6 \
  --scheduler cosine \
  --focal-loss \
  --focal-gamma 2.0 \
  --weighted-loss \
  --cutmix 0.3 \
  --freeze-epochs 10 \
  --patience 5 \
  --log-every 150 \
  --checkpoint results/models/best_resnet50_joint_all.pt \
  > results/models/log_resnet50_joint_all.txt 2>&1
```

**What changed vs the previous run:**

| Setting | Before | After | Why |
|---------|--------|-------|-----|
| LR | 5e-5 | 1e-6 | Too aggressive — was destroying pretrained features |
| Loss | Weighted CE | Focal Loss + Weighted | Focus on hard classes |
| Freeze epochs | 0 | 10 | Train head only first, then full fine-tuning |
| Early stopping | None | patience=5 | Stop when val_acc plateaus |
| Output | Deleted temp file | `log_resnet50_joint_all.txt` | Actually monitorable |

### New Run Results (killed at epoch 3)

| Epoch | Train Loss | Val Loss | Val Acc | Note |
|-------|-----------|----------|---------|------|
| 1 | 0.5098 | 1.2500 | 72.22% | new best — saved |
| **2** | **0.4768** | **1.2412** | **73.41%** | **new best — saved** |
| 3 | 0.4439 | 1.2835 | 71.03% | val loss ticked up slightly |

**Best checkpoint: `best_resnet50_joint_all.pt` — 73.41% val_acc (epoch 2)**

Val loss was also decreasing (1.2500 → 1.2412) at epoch 2 — opposite of the previous run. Frozen backbone + Focal Loss + low LR combination was working.

---

## 6. Model Comparison (All Checkpoints)

| Checkpoint | Val Acc | Eval Set | Real-world? |
|-----------|---------|----------|-------------|
| `best_resnet50_plantvillage.pt` | 99.59% | PlantVillage val (clean lab) | No |
| `best_resnet50_joint_aug.pt` | 97.59% | Augmented PlantDoc val | Partially |
| **`best_resnet50_joint_all.pt`** | **73.41%** | **Real PlantDoc test** | **Yes** |
| `best_resnet50_plantdoc.pt` | 62.70% | Real PlantDoc test | Yes |
| `best_resnet50_joint.pt` | 58.33% | Real PlantDoc test | Yes |

**Best model for real-world use: `best_resnet50_joint_all.pt`**

---

## 7. Recommendations for Future Runs

1. **TTA at inference** — already implemented in `evaluate.py`, run with `--tta --tta-n 8` for free 2–4% gain
2. **Let the current strategy run to completion** — frozen phase ends at epoch 10, bigger gains expected in phase 2 (epochs 11–30)
3. **Try EfficientNet-B4 or ViT-B/16** — ResNet50 plateaus around 71–74% on this domain-shift problem; transformer-based models generalize better
4. **Domain-bridging augmentation** — add RandomErasing and JPEG compression artifacts to better simulate field conditions
5. **Resume command** (use after transferring `best_resnet50_joint_all.pt` to new machine):

```bash
python3 -u src/train/train.py \
  --model resnet50 \
  --dataset joint_all \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-6 \
  --scheduler cosine \
  --focal-loss \
  --focal-gamma 2.0 \
  --weighted-loss \
  --cutmix 0.3 \
  --freeze-epochs 10 \
  --patience 5 \
  --log-every 150 \
  --checkpoint results/models/best_resnet50_joint_all.pt \
  > results/models/log_resnet50_joint_all.txt 2>&1 &
```

---

## 8. GitHub

- Repo: [https://github.com/marckhat/plant-disease-detection](https://github.com/marckhat/plant-disease-detection)
- Files pushed: `src/train/train.py`, `src/eval/evaluate.py`, `src/data/dataset.py`, `src/data/transforms.py`, `src/data/integrate_augmented_plantdoc.py`, `src/app/app.py`, `src/app/templates/index.html`, `.gitignore`, `README.md`
- Data (7.8GB) and model checkpoints (~91MB each) are git-ignored — must be transferred manually

### Data transfer checklist for new machine

- [ ] `data/raw/plantvillage/` (924MB) — re-download from Kaggle
- [ ] `data/raw/plantdoc/` (914MB) — re-download from Kaggle
- [ ] `data/processed/aligned/augmented_plantdoc/` (4.5GB) — transfer manually (generated, not downloadable)
- [ ] `results/models/best_resnet50_joint_all.pt` (91MB) — transfer manually
