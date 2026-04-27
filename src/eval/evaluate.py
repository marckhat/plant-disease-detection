"""Evaluate a trained classifier checkpoint on one aligned dataset split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Allow `python src/eval/evaluate.py` from repo root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import (
    load_plantdoc_test,
    load_plantdoc_train,
    load_plantvillage_val,
    make_eval_dataloader,
)
from src.models.model_factory import get_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--split",
        type=str,
        required=True,
        choices=("plantvillage_val", "plantdoc_train", "plantdoc_test"),
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    return p.parse_args()


def load_eval_dataset(split: str, image_size: int):
    """Load the requested split and return (dataset, class_names)."""
    if split == "plantvillage_val":
        return load_plantvillage_val(image_size=image_size)
    if split == "plantdoc_train":
        return load_plantdoc_train(image_size=image_size)
    if split == "plantdoc_test":
        return load_plantdoc_test(image_size=image_size)
    raise ValueError(
        f"Unsupported split {split!r}. "
        "Expected 'plantvillage_val', 'plantdoc_train', or 'plantdoc_test'."
    )


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run inference over one loader and collect true/pred label indices."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


def save_confusion_matrix_image(
    cm, class_names: list[str], out_path: Path, title: str
) -> None:
    """Save confusion matrix heatmap image."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset, class_names = load_eval_dataset(args.split, image_size=args.image_size)
    loader = make_eval_dataloader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = ckpt.get("model_name")
    if not model_name:
        raise ValueError(
            f"Checkpoint missing 'model_name': {checkpoint_path}. "
            "Expected a checkpoint saved by src/train/train.py."
        )

    model = get_model(model_name=model_name, num_classes=len(class_names), pretrained=False)
    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        raise ValueError(
            f"Checkpoint missing 'model_state_dict': {checkpoint_path}. "
            "Expected a checkpoint saved by src/train/train.py."
        )
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    y_true, y_pred = collect_predictions(model, loader, device)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )

    print("=" * 70)
    print("Evaluation results")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {model_name}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(y_true)}")
    print("-" * 70)
    print(f"Accuracy:          {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro):    {rec:.4f}")
    print(f"F1-score (macro):  {f1:.4f}")
    print("-" * 70)
    print("Classification report:")
    print(report_text)

    metrics_dir = _ROOT / "results" / "metrics"
    cm_dir = _ROOT / "results" / "confusion_matrices"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = str(model_name).strip().lower()
    metrics_path = metrics_dir / f"eval_{safe_model_name}_{args.split}.json"
    report_path = metrics_dir / f"report_{safe_model_name}_{args.split}.txt"
    cm_path = cm_dir / f"cm_{safe_model_name}_{args.split}.png"

    payload = {
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "split": args.split,
        "num_samples": len(y_true),
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    save_confusion_matrix_image(
        cm=cm,
        class_names=class_names,
        out_path=cm_path,
        title=f"Confusion Matrix - {model_name} - {args.split}",
    )

    print("-" * 70)
    print(f"Saved metrics JSON: {metrics_path}")
    print(f"Saved classification report text: {report_path}")
    print(f"Saved confusion matrix image: {cm_path}")


if __name__ == "__main__":
    main()

