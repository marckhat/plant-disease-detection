# plant-disease-detection

Computer vision **leaf image classification** system that predicts plant disease classes from RGB images and provides **high-level treatment or management recommendations** via simple rules.

## Scope

- **Training dataset**: PlantVillage (clean images)
- **Robustness evaluation**: PlantDoc (real-world images)
- **Models compared**: ResNet50, EfficientNet-B0 (PyTorch). During experimentation, **both models will be trained and evaluated**; **one final model will be selected** for deployment in the Streamlit demo.
- **Metrics**: accuracy, precision, recall, F1-score, confusion matrix
- **Explainability**: Grad-CAM visualizations
- **Demo**: simple Streamlit app using the selected final model

## Planned pipeline (high level)

1. Prepare data: organize datasets, create train/val/test splits, define transforms.
2. Train: train ResNet50 and EfficientNet-B0 on PlantVillage.
3. Evaluate: compute metrics + confusion matrix on PlantVillage splits and PlantDoc robustness set.
4. Explain: generate Grad-CAM examples for qualitative inspection.
5. Recommend: map predicted disease → high-level treatment or management recommendations (rule-based).
6. Demo: Streamlit UI that loads one final model for inference + Grad-CAM + recommendation text.