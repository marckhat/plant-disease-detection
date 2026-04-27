"""Train a pretrained classifier on aligned PlantVillage (train/val)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow `python src/train/train.py` from repo root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import load_all_datasets, make_eval_dataloader, make_train_dataloader
from src.models.model_factory import get_model
from src.utils.seed import set_seed


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """Freeze all params, then unfreeze the final classification layer only."""
    for p in model.parameters():
        p.requires_grad = False
    key = model_name.strip().lower()
    if key == "resnet50":
        for p in model.fc.parameters():
            p.requires_grad = True
    elif key == "efficientnet_b0":
        for p in model.classifier.parameters():
            p.requires_grad = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ResNet50 or EfficientNet-B0 on PlantVillage.")
    p.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=("resnet50", "efficientnet_b0"),
        help="Backbone architecture (default: resnet50).",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument(
        "--dataset",
        type=str,
        default="plantvillage",
        choices=("plantvillage", "plantdoc"),
        help="Dataset for fine-tuning/evaluation split selection.",
    )
    p.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Debug: stop training each epoch after this many batches (default: full pass).",
    )
    p.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Debug: stop validation each epoch after this many batches (default: full pass).",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Print batch progress every N batches in train/eval (0 = off).",
    )
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the classifier head; freeze pretrained backbone weights.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Optional early stopping patience based on validation accuracy.",
    )
    return p.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_train_batches: int | None = None,
    log_every: int = 20,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    n_batches = len(loader)
    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        if log_every > 0 and batch_idx % log_every == 0:
            avg_so_far = total_loss / max(n_samples, 1)
            print(
                f"[Train] Batch {batch_idx}/{n_batches} | "
                f"avg_loss_so_far={avg_so_far:.4f}"
            )

        if max_train_batches is not None and batch_idx >= max_train_batches:
            break
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    max_eval_batches: int | None = None,
    log_every: int = 20,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n_samples = 0
    n_batches = len(loader)
    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        bs = images.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        n_samples += bs

        if log_every > 0 and batch_idx % log_every == 0:
            print(f"[Eval] Batch {batch_idx}/{n_batches}")

        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break

    avg_loss = total_loss / max(n_samples, 1)
    accuracy = correct / max(n_samples, 1)
    return avg_loss, accuracy


def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    epoch: int,
    val_acc: float,
    model_name: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    image_size: int,
    freeze_backbone: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "model_name": model_name,
            "dataset": dataset,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "image_size": image_size,
            "freeze_backbone": freeze_backbone,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_dir = _ROOT / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = models_dir / f"best_{args.model}_{args.dataset}.pt"
    final_ckpt_path = models_dir / f"final_{args.model}_{args.dataset}.pt"

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    best_val_acc = -1.0
    epochs_without_improvement = 0

    pv_train, pv_val, pd_train, pd_test, class_names = load_all_datasets(
        image_size=args.image_size
    )
    num_classes = len(class_names)

    if args.dataset == "plantvillage":
        train_dataset = pv_train
        eval_dataset = pv_val
    else:
        train_dataset = pd_train
        eval_dataset = pd_test

    train_loader = make_train_dataloader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_loader = make_eval_dataloader(
        eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = get_model(args.model, num_classes, pretrained=True).to(device)
    model.train()

    if args.freeze_backbone:
        freeze_backbone(model, args.model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    print(f"Device: {device}")
    print(f"Model: {args.model} | Classes: {num_classes}")
    print(f"Dataset: {args.dataset}")
    print(f"Backbone freezing: {'enabled' if args.freeze_backbone else 'disabled'}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Learning rate: {args.lr}")
    print(f"Train size: {len(train_dataset)} | Val size: {len(eval_dataset)}")
    print(f"Sample classes: {class_names[:5]}")
    if args.patience is not None:
        print(f"Early stopping patience: {args.patience}")
    if args.max_train_batches is not None:
        print(
            f"Debug: max_train_batches={args.max_train_batches} "
            "(training stops after this many batches per epoch)"
        )
    if args.max_eval_batches is not None:
        print(
            f"Debug: max_eval_batches={args.max_eval_batches} "
            "(validation stops after this many batches per epoch)"
        )
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_train_batches=args.max_train_batches,
            log_every=args.log_every,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_eval_batches=args.max_eval_batches,
            log_every=args.log_every,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_ckpt_path,
                model,
                epoch=epoch,
                val_acc=val_acc,
                model_name=args.model,
                dataset=args.dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                image_size=args.image_size,
                freeze_backbone=args.freeze_backbone,
            )
            print(
                f"  New best val_acc={val_acc:.4f} — saved checkpoint: {best_ckpt_path}"
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.patience is not None and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping: no val_acc improvement for {args.patience} epoch(s)."
            )
            break

    save_checkpoint(
        final_ckpt_path,
        model,
        epoch=len(train_losses),
        val_acc=val_accuracies[-1] if val_accuracies else float("nan"),
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        freeze_backbone=args.freeze_backbone,
    )
    print(f"Final checkpoint saved: {final_ckpt_path}")

    history_path = models_dir / f"history_{args.model}_{args.dataset}.json"
    history = {
        "model_name": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "freeze_backbone": args.freeze_backbone,
    }
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")


if __name__ == "__main__":
    main()
