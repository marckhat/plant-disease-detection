# Verdis  ·  Plant Disease Detection

A PyTorch computer-vision pipeline for smartphone-based plant disease diagnosis, focused on **real-world robustness under domain shift**. The model is trained on the clean PlantVillage benchmark and evaluated on real field photographs from PlantDoc, with a Flask web dashboard for interactive inference, Grad-CAM explainability, and ResNet50 vs EfficientNet-B0 comparison.

**Final accuracy on real-world PlantDoc test set: 75.40% (TTA), F1 (macro) = 0.746, across 27 disease classes and 14 plant species.**

---

## Results

All evaluations on the 252-image **PlantDoc real-world test set** (smartphone-quality field photos):

| Model | Training data | Accuracy | F1 (macro) | Notes |
|-------|---------------|---------:|-----------:|-------|
| ResNet50 | PlantVillage only | ~30% | — | Lab-only baseline; collapses on field photos |
| ResNet50 | Joint (PV + PD) | 58.33% | — | + real-world data |
| ResNet50 | joint_all (PV + PD + aug PD) | 73.41% | — | + augmented PlantDoc |
| **ResNet50** | **joint_all + 8-view TTA** | **75.40%** | **0.7460** | **Production model** |
| EfficientNet-B0 | Joint (PV + PD) + TTA | 57.14% | 0.5535 | Architecture comparison |

Five of the 27 classes reach **perfect F1 = 1.00** (Grape healthy, Grape Black rot, Blueberry healthy, Pepper bell healthy, Strawberry healthy). Eleven classes reach F1 ≥ 0.85. The weakest classes — Potato Early/Late Blight and Corn Gray Leaf Spot — fail in clusters of visually similar pathologies.

> **Headline lesson**: data strategy outweighs architecture choice. ResNet50 and EfficientNet-B0 are within statistical noise of each other on identical conditions; the 18-point gap between the production model and either architecture trained on the smaller `joint` dataset is driven almost entirely by the additional augmented training data.

---

## Quick Start  ·  Inference

```bash
git clone https://github.com/marckhat/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt

# Place the pretrained checkpoint at:
#   results/models/best_resnet50_joint_all.pt
# (see "Pretrained weights" section below)

# Option A — launch the Flask dashboard
python src/app/app.py
# then open http://localhost:5000

# Option B — single-image inference via API
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/api/predict
```

Example response:

```json
{
  "predictions": [
    {"label": "Grape__Black_rot", "pretty": "Grape › Black rot", "probability": 0.8849,
     "info": {"common": "Grape Black Rot", "severity": "high",
              "treatment": "Remove mummified fruit. Apply fungicides..."}},
    {"label": "Grape__healthy",   "pretty": "Grape › healthy",   "probability": 0.0182, "info": {...}},
    "..."
  ]
}
```

---

## Features

| Capability | Where |
|---|---|
| **Domain-robust training** — PV + PD + augmented PD with strong augmentation | `src/train/train.py` |
| **Two-phase fine-tuning** — freeze backbone N epochs, then unfreeze (LR×0.1) | `--freeze-epochs N` |
| **Focal Loss** — γ = 2.0, with inverse-frequency class weights | `--focal-loss` |
| **CutMix regularization** | `--cutmix 0.3` |
| **Test-Time Augmentation** — 8 views averaged at inference | `--tta --tta-n 8` |
| **Grad-CAM explainability** — live heatmap overlay in dashboard | `/api/gradcam` |
| **Model comparison** — ResNet50 vs EfficientNet-B0 side-by-side | dashboard `Model comparison` view |
| **Evaluation dashboard** — confusion matrix + per-class F1 | dashboard `Evaluation` view |

---

## Flask Web Dashboard

```bash
python src/app/app.py
# → http://localhost:5000
```

The dashboard has six views:

| View | Description |
|---|---|
| **Diagnose** | Drop / pick a leaf photo, get top-5 predictions + confidence + severity + treatment |
| **Explainability** | Live Grad-CAM heatmap toggle (Original / Heatmap / Split) |
| **Model comparison** | ResNet50 vs EfficientNet-B0 metrics + per-class F1 chart |
| **Evaluation** | Confusion matrix + per-class F1, switchable between models |
| **Domain study** | The PV-only → joint → joint_all accuracy progression |
| **About & limits** | Hard classes, tech stack, API examples, credits |

### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | POST | Image → top-5 predictions + treatment |
| `/api/gradcam` | POST | Image → base64-encoded Grad-CAM overlay |
| `/api/metrics-all` | GET | Macro metrics for both ResNet50 and EfficientNet-B0 |
| `/api/per-class` | GET | Per-class F1 / precision / recall for both models |
| `/api/samples` | GET | Curated demo image gallery metadata |

---

## Project Structure

```
plant-disease-detection/
├── src/
│   ├── app/
│   │   ├── app.py                  # Flask dashboard + 5 API endpoints + Grad-CAM
│   │   ├── templates/index.html    # Single-page dark dashboard UI (Tailwind + Chart.js)
│   │   └── static/                 # Demo images + confusion matrices
│   ├── data/
│   │   ├── dataset.py              # ImageFolder loaders for all splits, class alignment
│   │   ├── transforms.py           # Eval / train / strong-aug torchvision pipelines
│   │   ├── build_aligned_dataset.py
│   │   ├── prepare_data.py
│   │   └── integrate_augmented_plantdoc.py
│   ├── eval/
│   │   └── evaluate.py             # Eval script with optional 8-view TTA
│   ├── models/
│   │   └── model_factory.py        # ResNet50 / EfficientNet-B0 factory
│   ├── train/
│   │   └── train.py                # Training script: focal loss, CutMix, 2-phase, early stop
│   └── utils/
│       └── seed.py
├── results/
│   ├── models/                     # Checkpoints (.pt files — git-ignored, ~91 MB each)
│   ├── metrics/                    # eval_<model>_<split>.json + classification reports
│   └── confusion_matrices/         # *.png confusion-matrix images
├── Project_Presentation.pptx       # 13-slide, 10-min talk deck
├── Project_Presentation_Full.pptx  # 23-slide comprehensive version
├── Project_Report.docx             # 11-page, 6-section academic report
├── requirements.txt
└── README.md
```

---

## Setup

### 1.  Clone

```bash
git clone https://github.com/marckhat/plant-disease-detection.git
cd plant-disease-detection
```

### 2.  Environment

```bash
python -m venv .venv
source .venv/bin/activate         # macOS / Linux
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

### 3.  Datasets

Download [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and [PlantDoc](https://www.kaggle.com/datasets/pratikkayal/plantdoc-dataset) from Kaggle and place them under:

```
data/raw/plantvillage/{train,val}/
data/raw/plantdoc/{train,test}/
```

Then build the aligned dataset (maps PlantDoc class names to PlantVillage's taxonomy and keeps the 27 shared classes):

```bash
python src/data/build_aligned_dataset.py
```

### 4.  Pretrained weights

The production checkpoint `best_resnet50_joint_all.pt` (≈91 MB) is **not** included in the git history due to size. You can obtain it two ways:

- **Reproduce training** — see the next section. End-to-end training takes ~10 hours on a single mid-range GPU.
- **Download from a release / shared link** — if you forked this repo, drop the file at `results/models/best_resnet50_joint_all.pt`.

---

## Reproducing the Production Model

The final training run uses a two-stage process: train on the augmented joint set, then fine-tune at very low LR with frozen-backbone warmup.

```bash
# Stage 1 — initial training on the full augmented dataset
python src/train/train.py \
  --model resnet50 \
  --dataset joint_all \
  --epochs 30 \
  --batch-size 32 \
  --lr 5e-5 \
  --scheduler cosine \
  --focal-loss --focal-gamma 2.0 \
  --weighted-loss \
  --cutmix 0.3 \
  --freeze-epochs 10 \
  --patience 5 \
  --log-every 150

# Stage 2 — gentle fine-tune from the saved checkpoint
python src/train/train.py \
  --model resnet50 \
  --dataset joint_all \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-6 \
  --scheduler cosine \
  --focal-loss --focal-gamma 2.0 \
  --weighted-loss \
  --cutmix 0.3 \
  --freeze-epochs 10 \
  --patience 5 \
  --log-every 150 \
  --checkpoint results/models/best_resnet50_joint_all.pt
```

### Training arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `resnet50` | `resnet50` or `efficientnet_b0` |
| `--dataset` | `plantvillage` | `plantvillage`, `plantdoc`, `joint`, `joint_aug`, `joint_all` |
| `--epochs` | 5 | Total training epochs |
| `--batch-size` | 32 | |
| `--lr` | 1e-4 | Learning rate (cosine-annealed if `--scheduler cosine`) |
| `--focal-loss` | off | Focal Loss instead of CrossEntropy |
| `--focal-gamma` | 2.0 | Focal Loss γ |
| `--weighted-loss` | off | Inverse-frequency class weights |
| `--cutmix` | 0.0 | CutMix α (0 = disabled) |
| `--freeze-epochs` | 0 | Freeze backbone for first N epochs, then unfreeze with LR×0.1 |
| `--patience` | none | Early stopping patience on val accuracy |
| `--checkpoint` | none | Resume / fine-tune from a saved `.pt` checkpoint |

---

## Evaluation

```bash
# Standard
python src/eval/evaluate.py \
  --checkpoint results/models/best_resnet50_joint_all.pt \
  --split plantdoc_test

# With Test-Time Augmentation (recommended)
python src/eval/evaluate.py \
  --checkpoint results/models/best_resnet50_joint_all.pt \
  --split plantdoc_test \
  --tta --tta-n 8
```

Outputs are written to:

```
results/metrics/eval_<model>_<split>.json     # accuracy, precision, recall, F1, classification report
results/metrics/report_<model>_<split>.txt    # human-readable classification report
results/confusion_matrices/cm_<model>_<split>.png
```

---

## Datasets

| Dataset | Images | Classes | Domain |
|---|---:|---:|---|
| [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) | ~54,000 | 27 | Controlled lab |
| [PlantDoc](https://www.kaggle.com/datasets/pratikkayal/plantdoc-dataset) | ~2,500 | 27 | Real-world field |

After alignment to 27 shared classes: 30,747 PV train / 7,372 PV val / 2,668 PD train / 252 PD test = **41,039 images total**.

Both datasets share 27 disease classes across 14 plant species (apple, blueberry, cherry, corn, grape, peach, pepper, potato, raspberry, soybean, squash, strawberry, tomato — plus healthy classes).

---

## Deliverables

This repository includes the full project deliverables:

- **`Project_Presentation.pptx`** — 13-slide deck designed for a 10-minute talk
- **`Project_Presentation_Full.pptx`** — 23-slide comprehensive version with full methodology depth
- **`Project_Report.docx`** — 11-page academic report (6 sections: Introduction, Dataset, Methodology, Experiments, Discussion, Conclusion)

---

## Notes

- `data/` and `results/` are git-ignored; datasets and checkpoints must be transferred separately.
- For reproducibility, all training runs use a fixed seed of 42.
- The dashboard is force-CPU at inference (see `_device = torch.device("cpu")` in `src/app/app.py`) to coexist with concurrent training; flip back to `cuda` for production speed if no training is running.
