"""ImageFolder-based datasets for aligned PlantVillage / PlantDoc splits."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .transforms import get_eval_transforms, get_train_transforms

# Default layout (under project root):
# data/processed/aligned/plantvillage/{train,val}/<class>/
# data/processed/aligned/plantdoc/{train,test}/<class>/


def project_root() -> Path:
    """Project root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def default_aligned_root(root: Path | None = None) -> Path:
    """Root folder containing ``plantvillage/`` and ``plantdoc/`` aligned splits."""
    return (root or project_root()) / "data" / "processed" / "aligned"


def _sorted_class_folder_names(split_root: Path) -> list[str]:
    """Class names = immediate subdirectories of ``split_root`` (ImageFolder-compatible)."""
    if not split_root.is_dir():
        raise FileNotFoundError(f"Expected a directory: {split_root}")
    names = [
        p.name
        for p in split_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]
    return sorted(names)


def reference_classes_and_idx(aligned_root: Path) -> Tuple[list[str], dict[str, int]]:
    """
    Use **PlantVillage train** as the canonical label space.

    Returns ``(class_names, class_to_idx)`` matching ``torchvision.datasets.ImageFolder``
    for that folder (sorted class directory names, indices 0..N-1).
    """
    pv_train = aligned_root / "plantvillage" / "train"
    classes = _sorted_class_folder_names(pv_train)
    if not classes:
        raise ValueError(
            f"No class folders found under PlantVillage train: {pv_train}. "
            "Build or populate data/processed/aligned/plantvillage/train first."
        )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx


def _assert_imagefolder_matches_reference(
    dataset: ImageFolder,
    ref_classes: list[str],
    ref_class_to_idx: dict[str, int],
    location: str,
) -> None:
    """Ensure ImageFolder used the same ``classes`` / ``class_to_idx`` as PlantVillage train."""
    if dataset.classes != ref_classes:
        raise ValueError(
            f"Class name list mismatch for {location}.\n"
            f"  Expected (PlantVillage train): {ref_classes}\n"
            f"  Found: {dataset.classes}"
        )
    if dataset.class_to_idx != ref_class_to_idx:
        raise ValueError(
            f"class_to_idx mismatch for {location}.\n"
            f"  Expected: {ref_class_to_idx}\n"
            f"  Found: {dataset.class_to_idx}"
        )


def _make_imagefolder(
    split_root: Path,
    transform,
    ref_classes: list[str],
    ref_class_to_idx: dict[str, int],
    location: str,
) -> ImageFolder:
    """Load ``ImageFolder`` and verify it matches the PlantVillage train reference."""
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_root}")

    on_disk = _sorted_class_folder_names(split_root)
    if on_disk != ref_classes:
        raise ValueError(
            f"Class folders under {location} do not match PlantVillage train.\n"
            f"  Expected (sorted): {ref_classes}\n"
            f"  Found (sorted):    {on_disk}"
        )

    dataset = ImageFolder(root=str(split_root), transform=transform)
    _assert_imagefolder_matches_reference(
        dataset, ref_classes, ref_class_to_idx, location
    )
    return dataset


def load_plantvillage_train(
    aligned_root: Path | None = None,
    *,
    transform=None,
    image_size: int = 224,
) -> Tuple[ImageFolder, list[str]]:
    """PlantVillage **train** split; defines global ``classes`` / ``class_to_idx``."""
    root = default_aligned_root(aligned_root)
    ref_classes, ref_idx = reference_classes_and_idx(root)
    tfm = transform if transform is not None else get_train_transforms(image_size)
    ds = _make_imagefolder(
        root / "plantvillage" / "train",
        tfm,
        ref_classes,
        ref_idx,
        "plantvillage/train",
    )
    return ds, ref_classes


def load_plantvillage_val(
    aligned_root: Path | None = None,
    *,
    transform=None,
    image_size: int = 224,
) -> Tuple[ImageFolder, list[str]]:
    """PlantVillage **val** split (eval transforms by default)."""
    root = default_aligned_root(aligned_root)
    ref_classes, ref_idx = reference_classes_and_idx(root)
    tfm = transform if transform is not None else get_eval_transforms(image_size)
    ds = _make_imagefolder(
        root / "plantvillage" / "val",
        tfm,
        ref_classes,
        ref_idx,
        "plantvillage/val",
    )
    return ds, ref_classes


def load_plantdoc_train(
    aligned_root: Path | None = None,
    *,
    transform=None,
    image_size: int = 224,
) -> Tuple[ImageFolder, list[str]]:
    """PlantDoc **train** split (eval transforms by default — no augmentation here)."""
    root = default_aligned_root(aligned_root)
    ref_classes, ref_idx = reference_classes_and_idx(root)
    tfm = transform if transform is not None else get_eval_transforms(image_size)
    ds = _make_imagefolder(
        root / "plantdoc" / "train",
        tfm,
        ref_classes,
        ref_idx,
        "plantdoc/train",
    )
    return ds, ref_classes


def load_plantdoc_test(
    aligned_root: Path | None = None,
    *,
    transform=None,
    image_size: int = 224,
) -> Tuple[ImageFolder, list[str]]:
    """PlantDoc **test** split."""
    root = default_aligned_root(aligned_root)
    ref_classes, ref_idx = reference_classes_and_idx(root)
    tfm = transform if transform is not None else get_eval_transforms(image_size)
    ds = _make_imagefolder(
        root / "plantdoc" / "test",
        tfm,
        ref_classes,
        ref_idx,
        "plantdoc/test",
    )
    return ds, ref_classes


def _make_imagefolder_subset(
    split_root: Path,
    transform,
    ref_class_to_idx: dict[str, int],
    location: str,
) -> ImageFolder:
    """Load ImageFolder whose classes are a subset of the reference; remap indices to global."""
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_root}")

    on_disk = _sorted_class_folder_names(split_root)
    unknown = [c for c in on_disk if c not in ref_class_to_idx]
    if unknown:
        raise ValueError(f"{location}: class folders not in reference: {unknown}")

    local_to_global = {i: ref_class_to_idx[c] for i, c in enumerate(sorted(on_disk))}
    dataset = ImageFolder(
        root=str(split_root),
        transform=transform,
        target_transform=lambda y, m=local_to_global: m[y],
    )
    return dataset


def load_augmented_plantdoc_train(
    aligned_root: Path | None = None,
    *,
    transform=None,
    image_size: int = 224,
) -> Tuple[ImageFolder, list[str]]:
    """Augmented PlantDoc **train** split (27 classes, remapped to global indices)."""
    root = default_aligned_root(aligned_root)
    ref_classes, ref_idx = reference_classes_and_idx(root)
    from .transforms import get_strong_train_transforms
    tfm = transform if transform is not None else get_strong_train_transforms(image_size)
    ds = _make_imagefolder_subset(
        root / "augmented_plantdoc" / "train", tfm, ref_idx, "augmented_plantdoc/train"
    )
    return ds, ref_classes


def load_augmented_plantdoc_val(
    aligned_root: Path | None = None,
    *,
    transform=None,
    image_size: int = 224,
) -> Tuple[ImageFolder, list[str]]:
    """Augmented PlantDoc **val** split (27 classes, remapped to global indices)."""
    root = default_aligned_root(aligned_root)
    ref_classes, ref_idx = reference_classes_and_idx(root)
    tfm = transform if transform is not None else get_eval_transforms(image_size)
    ds = _make_imagefolder_subset(
        root / "augmented_plantdoc" / "val", tfm, ref_idx, "augmented_plantdoc/val"
    )
    return ds, ref_classes


def load_augmented_plantdoc_test(
    aligned_root: Path | None = None,
    *,
    transform=None,
    image_size: int = 224,
) -> Tuple[ImageFolder, list[str]]:
    """Augmented PlantDoc **test** split (27 classes, remapped to global indices)."""
    root = default_aligned_root(aligned_root)
    ref_classes, ref_idx = reference_classes_and_idx(root)
    tfm = transform if transform is not None else get_eval_transforms(image_size)
    ds = _make_imagefolder_subset(
        root / "augmented_plantdoc" / "test", tfm, ref_idx, "augmented_plantdoc/test"
    )
    return ds, ref_classes


def load_all_datasets(
    aligned_root: Path | None = None,
    *,
    train_transform=None,
    eval_transform=None,
    image_size: int = 224,
) -> Tuple[ImageFolder, ImageFolder, ImageFolder, ImageFolder, list[str]]:
    """
    Load PlantVillage train/val and PlantDoc train/test with one reference ``class_names``.

    Uses training augmentations for PlantVillage train and eval transforms for the other
    splits unless custom transforms are supplied.
    """
    pv_train, class_names = load_plantvillage_train(
        aligned_root, transform=train_transform, image_size=image_size
    )
    pv_val, _ = load_plantvillage_val(
        aligned_root, transform=eval_transform, image_size=image_size
    )
    pd_train, _ = load_plantdoc_train(
        aligned_root, transform=eval_transform, image_size=image_size
    )
    pd_test, _ = load_plantdoc_test(
        aligned_root, transform=eval_transform, image_size=image_size
    )
    return pv_train, pv_val, pd_train, pd_test, class_names


def load_all_datasets_with_augmented(
    aligned_root: Path | None = None,
    *,
    train_transform=None,
    eval_transform=None,
    image_size: int = 224,
) -> Tuple[ImageFolder, ImageFolder, ImageFolder, ImageFolder, ImageFolder, list[str]]:
    """
    Load PlantVillage train/val, PlantDoc train/test, AND augmented_plantdoc train with one reference ``class_names``.

    Returns: (pv_train, pv_val, pd_train, aug_train, pd_test, class_names)
    """
    pv_train, class_names = load_plantvillage_train(
        aligned_root, transform=train_transform, image_size=image_size
    )
    pv_val, _ = load_plantvillage_val(
        aligned_root, transform=eval_transform, image_size=image_size
    )
    pd_train, _ = load_plantdoc_train(
        aligned_root, transform=eval_transform, image_size=image_size
    )
    aug_train, _ = load_augmented_plantdoc_train(
        aligned_root, transform=train_transform, image_size=image_size
    )
    pd_test, _ = load_plantdoc_test(
        aligned_root, transform=eval_transform, image_size=image_size
    )
    return pv_train, pv_val, pd_train, aug_train, pd_test, class_names


def make_dataloader(
    dataset: ImageFolder,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """
    Wrap ``dataset`` in a ``DataLoader``.

    Use ``shuffle=True`` for training only; validation / test should use ``shuffle=False``.
    ``pin_memory`` is enabled when CUDA is available.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_train_dataloader(
    dataset: ImageFolder,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """Training loader with shuffling enabled."""
    return make_dataloader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )


def make_eval_dataloader(
    dataset: ImageFolder,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """Validation / evaluation loader without shuffling."""
    return make_dataloader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
