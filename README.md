# Plant Disease Detection

PyTorch-based computer vision project for plant disease image classification from leaf photos, with a focus on robust evaluation under real-world domain shift.

## Project Overview

- **Training dataset (source domain):** PlantVillage (clean, lab-like images)
- **Evaluation dataset (target domain):** PlantDoc (real-world field images)
- **Model comparison:** ResNet50 vs EfficientNet-B0
- **Final model selection:** both models are trained/evaluated, then one is selected for deployment

## Domain Shift and Fine-Tuning

The project explicitly studies **domain shift**: models trained on PlantVillage often perform worse on PlantDoc due to background clutter, lighting variation, and image quality differences.  
To address this, the training pipeline supports **fine-tuning on PlantDoc splits** while preserving a consistent class mapping across datasets.

## Setup

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd plant-disease-detection
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Place datasets

Put raw datasets under:

```text
data/raw/
  plantvillage/
  plantdoc/
```

Expected split-first structure:

```text
data/raw/plantvillage/
  train/
  val/

data/raw/plantdoc/
  train/
  test/
```

## Notes

- `data/` and `results/` are git-ignored to keep repository pushes clean.
- Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`) are excluded from version control.