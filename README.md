# Plant Disease Detection

A PyTorch computer vision pipeline for classifying plant diseases from leaf photographs, with a focus on **real-world robustness** under domain shift. Models are trained on the clean PlantVillage dataset and evaluated on the challenging real-world PlantDoc benchmark.

---

## Results

| Model | Dataset | Val Accuracy | Eval Set |
|-------|---------|-------------|----------|
| ResNet50 | PlantVillage only | 99.59% | PlantVillage val (clean) |
| ResNet50 | Joint + Augmented (joint_aug) | 97.59% | Augmented PlantDoc val |
| **ResNet50** | **All sources (joint_all)** | **73.41%** | **Real PlantDoc test** |
| ResNet50 | PlantDoc only | 62.70% | Real PlantDoc test |
| ResNet50 | Joint (PV + PD) | 58.33% | Real PlantDoc test |

> The `joint_all` model is the recommended checkpoint for real-world deployment — it is the only one trained on all data sources and evaluated on actual field photographs.

---

## Features

- **Two-phase fine-tuning** — freeze backbone for N epochs (head-only), then unfreeze for full fine-tuning
- **Focal Loss** — down-weights easy examples to force focus on hard classes (Potato blight, Corn gray leaf spot, Tomato bacterial spot)
- **CutMix augmentation** — mixes patches between images for better generalization
- **Weighted loss** — inverse-frequency class weights to handle class imbalance
- **Test-Time Augmentation (TTA)** — averages predictions over 8 augmented views at inference
- **Early stopping** — stops training when validation accuracy plateaus
- **Grad-CAM** — visualizes which leaf regions the model attends to
- **Flask web app** — upload a leaf photo and get an instant disease prediction with treatment recommendations

---

## Project Structure

```
plant-disease-detection/
├── src/
│   ├── app/
│   │   ├── app.py                        # Flask web dashboard
│   │   └── templates/index.html          # Frontend UI
│   ├── data/
│   │   ├── dataset.py                    # Dataset loaders for all splits
│   │   ├── transforms.py                 # Train / eval / strong augmentation pipelines
│   │   ├── build_aligned_dataset.py      # Aligns PlantVillage & PlantDoc class labels
│   │   ├── prepare_data.py               # Raw data preprocessing
│   │   └── integrate_augmented_plantdoc.py
│   ├── eval/
│   │   └── evaluate.py                   # Evaluation script with TTA support
│   ├── models/
│   │   ├── model_factory.py              # ResNet50 / EfficientNet-B0 factory
│   │   ├── gradcam.py                    # Grad-CAM visualizations
│   │   └── predict.py                    # Single-image inference
│   ├── train/
│   │   └── train.py                      # Main training script
│   ├── support/
│   │   └── recommendations.py            # Disease treatment recommendations
│   └── utils/
│       ├── seed.py                       # Reproducibility
│       ├── metrics.py
│       └── plotting.py
├── data/                                 # Git-ignored — place datasets here
│   ├── raw/
│   │   ├── plantvillage/
│   │   └── plantdoc/
│   └── processed/aligned/
├── results/                              # Git-ignored — checkpoints & metrics
│   ├── models/
│   └── metrics/
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/marckhat/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\Activate.ps1       # Windows PowerShell
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place datasets

Download [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and [PlantDoc](https://www.kaggle.com/datasets/pratikkayal/plantdoc-dataset) from Kaggle and place them under:

```
data/raw/plantvillage/
    train/
    val/

data/raw/plantdoc/
    train/
    test/
```

Then build the aligned dataset:

```bash
python src/data/build_aligned_dataset.py
```

---

## Training

### Recommended run (best real-world accuracy)

```bash
python src/train/train.py \
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
  --checkpoint results/models/best_resnet50_joint_all.pt
```

### Key training arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | `plantvillage`, `plantdoc`, `joint`, `joint_aug`, `joint_all` |
| `--focal-loss` | Use Focal Loss instead of CrossEntropy |
| `--focal-gamma` | Focal Loss gamma (default: 2.0) |
| `--weighted-loss` | Inverse-frequency class weights |
| `--freeze-epochs N` | Freeze backbone for first N epochs |
| `--patience N` | Early stopping patience |
| `--cutmix` | CutMix alpha (0 = disabled) |
| `--checkpoint` | Resume / fine-tune from a saved checkpoint |

---

## Evaluation

### Standard evaluation

```bash
python src/eval/evaluate.py \
  --checkpoint results/models/best_resnet50_joint_all.pt \
  --split plantdoc_test
```

### With Test-Time Augmentation (TTA)

```bash
python src/eval/evaluate.py \
  --checkpoint results/models/best_resnet50_joint_all.pt \
  --split plantdoc_test \
  --tta \
  --tta-n 8
```

TTA averages predictions over 8 augmented views (flips, rotations, crops) and typically adds 2–4% accuracy with no retraining.

---

## Flask Web App

```bash
python src/app/app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser. Upload a leaf photo to get:
- Predicted disease class
- Confidence score
- Treatment recommendations
- Top-5 alternative predictions

---

## Datasets

| Dataset | Images | Classes | Domain |
|---------|--------|---------|--------|
| [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) | ~54,000 | 27 | Controlled lab |
| [PlantDoc](https://www.kaggle.com/datasets/pratikkayal/plantdoc-dataset) | ~2,500 | 27 | Real-world field |

Both datasets share 27 aligned disease classes across 14 plant species.

---

## Notes

- `data/` and `results/` are git-ignored — datasets and model checkpoints must be transferred separately.
- Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`) are excluded from version control due to size (~91MB each).
- For reproducibility, all runs use a fixed seed of 42.
