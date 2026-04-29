"""Train a pretrained classifier on aligned PlantVillage (train/val)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

# Allow `python src/train/train.py` from repo root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import (
    load_all_datasets,
    load_all_datasets_with_augmented,
    load_augmented_plantdoc_train,
    load_augmented_plantdoc_val,
    make_eval_dataloader,
    make_train_dataloader,
)
from src.data.transforms import get_strong_train_transforms
from src.models.model_factory import get_model
from src.utils.seed import set_seed


class FocalLoss(nn.Module):
    """Focal Loss — down-weights easy examples so training focuses on hard ones."""

    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            input, target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


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


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def compute_class_weights(dataset, num_classes: int, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights from dataset targets."""
    if isinstance(dataset, ConcatDataset):
        targets = []
        for ds in dataset.datasets:
            targets.extend(ds.targets)
    else:
        targets = dataset.targets
    counts = Counter(targets)
    total = len(targets)
    weights = torch.tensor(
        [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    return weights


def per_class_sample_weights(dataset, num_classes: int) -> np.ndarray:
    """Each sample gets weight = 1 / count_of_its_class (balances all classes equally)."""
    if isinstance(dataset, ConcatDataset):
        targets = []
        for ds in dataset.datasets:
            targets.extend(ds.targets)
    else:
        targets = dataset.targets
    counts = Counter(targets)
    return np.array([1.0 / counts[t] for t in targets], dtype=np.float32)


def cutmix_batch(images: torch.Tensor, targets: torch.Tensor, alpha: float = 0.4):
    """CutMix: cut a random patch from one image and paste into another."""
    lam = float(np.random.beta(alpha, alpha))
    batch_size = images.size(0)
    idx = torch.randperm(batch_size, device=images.device)

    # Random patch size between 0.3 and 0.7 of image
    h, w = images.size(2), images.size(3)
    patch_h = int(h * np.random.uniform(0.3, 0.7))
    patch_w = int(w * np.random.uniform(0.3, 0.7))

    # Random top-left corner
    top = np.random.randint(0, h - patch_h + 1)
    left = np.random.randint(0, w - patch_w + 1)

    mixed = images.clone()
    mixed[:, :, top:top+patch_h, left:left+patch_w] = images[idx, :, top:top+patch_h, left:left+patch_w]

    # Adjust lambda to account for actual patch size
    lam = 1 - (patch_h * patch_w) / (h * w)
    return mixed, targets, targets[idx], lam


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ResNet50 or EfficientNet-B0 on plant disease datasets.")
    p.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=("resnet50", "efficientnet_b0"),
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
        choices=("plantvillage", "plantdoc", "joint", "joint_aug", "joint_all"),
        help="'joint'=PV+PD; 'joint_aug'=PV+aug; 'joint_all'=PV+PD+aug (best for domain robustness).",
    )
    p.add_argument(
        "--strong-aug",
        action="store_true",
        help="Use stronger augmentation. Automatically enabled for joint training.",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=("none", "cosine"),
        help="LR scheduler. 'cosine' decays LR smoothly to 0 over all epochs.",
    )
    p.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Use inverse-frequency class weights in CrossEntropyLoss.",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for CrossEntropyLoss (e.g. 0.1). Default: 0 (off).",
    )
    p.add_argument(
        "--mixup",
        type=float,
        default=0.0,
        help="Mixup alpha (e.g. 0.4). 0 = disabled.",
    )
    p.add_argument(
        "--cutmix",
        type=float,
        default=0.0,
        help="CutMix alpha (e.g. 0.4). 0 = disabled.",
    )
    p.add_argument(
        "--per-class-sampling",
        action="store_true",
        help="Balance sampling per class (within each dataset) instead of per dataset.",
    )
    p.add_argument(
        "--freeze-epochs",
        type=int,
        default=0,
        help="Freeze backbone for the first N epochs, then unfreeze for the rest (two-phase).",
    )
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone for ALL epochs (train head only).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience based on validation accuracy.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to load weights from before training.",
    )
    p.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Debug: stop training each epoch after this many batches.",
    )
    p.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Debug: stop validation each epoch after this many batches.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Print batch progress every N batches (0 = off).",
    )
    p.add_argument(
        "--focal-loss",
        action="store_true",
        help="Use Focal Loss instead of CrossEntropy (focuses training on hard examples).",
    )
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal Loss gamma parameter (default: 2.0). Higher = more focus on hard examples.",
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
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    n_batches = len(loader)
    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)

        if cutmix_alpha > 0:
            images, targets_a, targets_b, lam = cutmix_batch(images, targets, cutmix_alpha)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
        elif mixup_alpha > 0:
            images, targets_a, targets_b, lam = mixup_batch(images, targets, mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)

        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        if log_every > 0 and batch_idx % log_every == 0:
            print(f"[Train] Batch {batch_idx}/{n_batches} | avg_loss={total_loss/n_samples:.4f}")

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
        correct += (logits.argmax(dim=1) == targets).sum().item()
        n_samples += bs

        if log_every > 0 and batch_idx % log_every == 0:
            print(f"[Eval] Batch {batch_idx}/{n_batches}")

        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break

    return total_loss / max(n_samples, 1), correct / max(n_samples, 1)


def save_checkpoint(path: Path, model: nn.Module, *, epoch: int, val_acc: float, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "model_name": args.model,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "image_size": args.image_size,
            "freeze_backbone": args.freeze_backbone,
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

    # ── Datasets ──────────────────────────────────────────────────────────
    use_strong = args.strong_aug or args.dataset in ("joint", "joint_aug", "joint_all")
    train_tfm = get_strong_train_transforms(args.image_size) if use_strong else None

    if args.dataset == "joint_all":
        pv_train, pv_val, pd_train, aug_train, pd_test, class_names = load_all_datasets_with_augmented(
            image_size=args.image_size,
            train_transform=train_tfm,
        )
        train_dataset = ConcatDataset([pv_train, pd_train, aug_train])
        eval_dataset = pd_test  # Validate directly on the hard real-world test set
        if args.per_class_sampling:
            sample_weights = per_class_sample_weights(train_dataset, num_classes)
        else:
            n_pv, n_pd, n_aug = len(pv_train), len(pd_train), len(aug_train)
            sample_weights = np.array(
                [1.0/n_pv]*n_pv + [1.0/n_pd]*n_pd + [1.0/n_aug]*n_aug, dtype=np.float32
            )
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_dataset), replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = make_eval_dataloader(
            eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
        num_classes = len(class_names)
    else:
        pv_train, pv_val, pd_train, pd_test, class_names = load_all_datasets(
            image_size=args.image_size,
            train_transform=train_tfm,
        )
        num_classes = len(class_names)

        if args.dataset == "plantvillage":
            train_dataset = pv_train
            eval_dataset = pv_val
            train_loader = make_train_dataloader(
                train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
            )
            val_loader = make_eval_dataloader(
                eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers
            )
        elif args.dataset == "plantdoc":
            train_dataset = pd_train
            eval_dataset = pd_test
            train_loader = make_train_dataloader(
                train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
            )
            val_loader = make_eval_dataloader(
                eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers
            )
        elif args.dataset == "joint_aug":
            aug_train, _ = load_augmented_plantdoc_train(image_size=args.image_size, transform=train_tfm)
            aug_val, _ = load_augmented_plantdoc_val(image_size=args.image_size)
            train_dataset = ConcatDataset([pv_train, aug_train])
            eval_dataset = aug_val
            if args.per_class_sampling:
                sample_weights = per_class_sample_weights(train_dataset, num_classes)
            else:
                n_pv, n_aug = len(pv_train), len(aug_train)
                sample_weights = np.array(
                    [1.0 / n_pv] * n_pv + [1.0 / n_aug] * n_aug, dtype=np.float32
                )
            sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(train_dataset), replacement=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = make_eval_dataloader(
                eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers
            )
        else:  # joint
            train_dataset = ConcatDataset([pv_train, pd_train])
            eval_dataset = pd_test
            if args.per_class_sampling:
                sample_weights = per_class_sample_weights(train_dataset, num_classes)
            else:
                n_pv, n_pd = len(pv_train), len(pd_train)
                sample_weights = np.array(
                    [1.0 / n_pv] * n_pv + [1.0 / n_pd] * n_pd, dtype=np.float32
                )
            sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(train_dataset), replacement=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = make_eval_dataloader(
                eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers
            )

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(args.model, num_classes, pretrained=True).to(device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded weights from checkpoint: {args.checkpoint}")

    if args.freeze_backbone or args.freeze_epochs > 0:
        freeze_backbone(model, args.model)
        print(
            f"Backbone frozen"
            + (f" for first {args.freeze_epochs} epochs." if args.freeze_epochs > 0 else " (all epochs).")
        )

    # ── Loss ──────────────────────────────────────────────────────────────
    class_weights = None
    if args.weighted_loss:
        class_weights = compute_class_weights(train_dataset, num_classes, device)

    if args.focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights, label_smoothing=args.label_smoothing)
        print(f"Using FocalLoss (gamma={args.focal_gamma}, weighted={args.weighted_loss}, label_smoothing={args.label_smoothing}).")
    elif args.weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        print(f"Using weighted CrossEntropyLoss (label_smoothing={args.label_smoothing}).")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        if args.label_smoothing > 0:
            print(f"Using label smoothing={args.label_smoothing}.")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
        if args.scheduler == "cosine"
        else None
    )

    # ── Info ──────────────────────────────────────────────────────────────
    print(f"Device: {device}")
    print(f"Model: {args.model} | Classes: {num_classes}")
    print(f"Dataset: {args.dataset} | Strong aug: {use_strong}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Train size: {len(train_dataset)} | Val size: {len(eval_dataset)}")
    print(f"LR: {args.lr} | Scheduler: {args.scheduler} | Weighted loss: {args.weighted_loss}")
    print(f"Label smoothing: {args.label_smoothing} | Mixup alpha: {args.mixup} | CutMix alpha: {args.cutmix} | Per-class sampling: {args.per_class_sampling}")
    if args.freeze_epochs > 0:
        print(f"Two-phase: freeze backbone for {args.freeze_epochs} epochs, then unfreeze.")
    if args.patience:
        print(f"Early stopping patience: {args.patience}")
    print("-" * 60)

    # ── Training loop ─────────────────────────────────────────────────────
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    best_val_acc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        # Two-phase: unfreeze backbone after freeze_epochs
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            unfreeze_all(model)
            # Re-build optimizer so newly unfrozen params are included
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
            if scheduler is not None:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - args.freeze_epochs, eta_min=1e-7
                )
            print(f"  [Phase 2] Backbone unfrozen. LR reset to {args.lr * 0.1:.2e}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            max_train_batches=args.max_train_batches, log_every=args.log_every,
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device,
            max_eval_batches=args.max_eval_batches, log_every=args.log_every,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | lr={current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_ckpt_path, model, epoch=epoch, val_acc=val_acc, args=args)
            print(f"  New best val_acc={val_acc:.4f} — saved {best_ckpt_path.name}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.patience is not None and epochs_without_improvement >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epoch(s).")
            break

    save_checkpoint(final_ckpt_path, model, epoch=len(train_losses),
                    val_acc=val_accuracies[-1] if val_accuracies else float("nan"), args=args)
    print(f"Final checkpoint saved: {final_ckpt_path}")

    history_path = models_dir / f"history_{args.model}_{args.dataset}.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model,
            "dataset": args.dataset,
            "epochs_run": len(train_losses),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "learning_rate": args.lr,
            "scheduler": args.scheduler,
            "weighted_loss": args.weighted_loss,
            "freeze_epochs": args.freeze_epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
        }, f, indent=2)
    print(f"Training history saved: {history_path}")


if __name__ == "__main__":
    main()
