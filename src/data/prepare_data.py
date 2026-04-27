"""Inspect and validate class-folder layout for PlantVillage and PlantDoc before training."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# Recognized split folder names at dataset root (case-insensitive).
SPLIT_NAMES = frozenset({"train", "val", "valid", "test"})
_SPLIT_PRINT_ORDER = ["train", "val", "valid", "test"]


def project_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def default_raw_paths(root: Path | None = None) -> tuple[Path, Path]:
    r = root or project_root()
    return (
        r / "data" / "raw" / "plantvillage",
        r / "data" / "raw" / "plantdoc",
    )


def count_images_in_class_folder(class_dir: Path) -> int:
    """Count image files under ``class_dir`` (recursive), using common extensions."""
    n = 0
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            n += 1
    return n


def _is_split_dir_name(name: str) -> bool:
    return name.lower() in SPLIT_NAMES


def _sort_split_keys(split_keys: list[str]) -> list[str]:
    def sort_key(name: str) -> tuple[int, str]:
        low = name.lower()
        try:
            pos = _SPLIT_PRINT_ORDER.index(low)
        except ValueError:
            pos = len(_SPLIT_PRINT_ORDER)
        return (pos, name.lower())

    return sorted(split_keys, key=sort_key)


def _map_class_subfolders_to_counts(parent: Path) -> dict[str, int]:
    """Each immediate child directory is a class; value is image count under that folder."""
    out: dict[str, int] = {}
    for p in parent.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            out[p.name] = count_images_in_class_folder(p)
    return out


@dataclass(frozen=True)
class DatasetLayout:
    """Result of inspecting one dataset root (must exist and be a directory)."""

    root: Path
    structure: Literal["direct", "split_first"]
    # Top-level dirs that look like class folders but were not scanned (split-first only).
    ignored_top_level: tuple[str, ...]
    # direct: class -> image count
    direct_counts: dict[str, int] | None
    # split_first: split folder name -> (class -> image count)
    per_split_counts: dict[str, dict[str, int]] | None

    def class_union(self) -> set[str]:
        if self.structure == "direct":
            assert self.direct_counts is not None
            return set(self.direct_counts)
        assert self.per_split_counts is not None
        u: set[str] = set()
        for counts in self.per_split_counts.values():
            u |= set(counts)
        return u


def inspect_dataset_layout(dataset_root: Path) -> DatasetLayout | None:
    """
    Detect direct-class vs split-first layout and collect per-class image counts.

    * **direct**: ``dataset_root/<class>/`` contains images.
    * **split-first**: ``dataset_root/<train|val|valid|test>/<class>/`` contains images.

    If at least one immediate subdirectory name matches a split name (case-insensitive),
    the layout is treated as split-first; only those split directories are scanned.
    Any other top-level directories are reported in ``ignored_top_level`` with a warning.
    """
    if not dataset_root.exists():
        print(f"WARNING: Missing dataset directory (skipped):\n  {dataset_root}")
        return None
    if not dataset_root.is_dir():
        print(f"WARNING: Path is not a directory (skipped):\n  {dataset_root}")
        return None

    children = [p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    split_dirs = [p for p in children if _is_split_dir_name(p.name)]
    other_dirs = [p for p in children if not _is_split_dir_name(p.name)]

    if split_dirs:
        if other_dirs:
            ignored = tuple(sorted(p.name for p in other_dirs))
            print(
                "WARNING: Split-first layout detected; ignoring non-split top-level folder(s):\n"
                f"  {', '.join(ignored)}"
            )
        per_split: dict[str, dict[str, int]] = {}
        for sd in split_dirs:
            per_split[sd.name] = _map_class_subfolders_to_counts(sd)
        return DatasetLayout(
            root=dataset_root,
            structure="split_first",
            ignored_top_level=tuple(sorted(p.name for p in other_dirs)),
            direct_counts=None,
            per_split_counts=per_split,
        )

    direct = _map_class_subfolders_to_counts(dataset_root)
    return DatasetLayout(
        root=dataset_root,
        structure="direct",
        ignored_top_level=(),
        direct_counts=direct,
        per_split_counts=None,
    )


def create_filtered_class_folders(
    source_root: Path,
    destination_root: Path,
    class_names: set[str],
    *,
    overwrite: bool = False,
) -> None:
    """
    Copy only class subfolders whose names appear in ``class_names`` from
    ``source_root`` into ``destination_root`` (same relative folder names).

    Expects a **direct** class-folder layout under ``source_root``. Not run by default;
    call explicitly when you want a reduced copy aligned on common labels.
    Creates ``destination_root`` if needed.
    """
    destination_root.mkdir(parents=True, exist_ok=True)
    for name in sorted(class_names):
        src = source_root / name
        if not src.is_dir():
            continue
        dst = destination_root / name
        if dst.exists():
            if overwrite:
                shutil.rmtree(dst)
            else:
                print(f"WARNING: Skip existing (use overwrite=True): {dst}")
                continue
        shutil.copytree(src, dst)


def _print_class_list(title: str, names: set[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not names:
        print("  (none)")
        return
    for n in sorted(names):
        print(f"  {n}")


def _print_class_counts_lines(class_counts: dict[str, int], indent: str = "  ") -> None:
    if not class_counts:
        print(f"{indent}(no class folders)")
        return
    for name in sorted(class_counts):
        c = class_counts[name]
        img_word = "image" if c == 1 else "images"
        print(f"{indent}{name}  ({c} {img_word})")


def _print_layout_details(label: str, layout: DatasetLayout) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(f"  Path: {layout.root}")
    if layout.structure == "direct":
        print("  Detected structure: direct-class (class folders under dataset root)")
        print("  Splits: (none — single pool)")
        n_classes = len(layout.direct_counts or {})
        print(f"  Number of classes: {n_classes}")
        print("  Class names and image counts:")
        _print_class_counts_lines(layout.direct_counts or {}, indent="    ")
        return

    print("  Detected structure: split-first (train / val / valid / test under dataset root)")
    assert layout.per_split_counts is not None
    split_keys = _sort_split_keys(list(layout.per_split_counts.keys()))
    print(f"  Available splits: {', '.join(split_keys)}")
    union = layout.class_union()
    print(f"  Union of classes across splits: {len(union)} class(es)")
    print("  Class names (union):")
    if not union:
        print("    (none)")
    else:
        for n in sorted(union):
            print(f"    {n}")
    print("  Image counts per class, per split:")
    for sk in split_keys:
        counts = layout.per_split_counts[sk]
        print(f"    [{sk}]")
        _print_class_counts_lines(counts, indent="      ")


def main() -> None:
    root = project_root()
    pv_root, pd_root = default_raw_paths(root)

    print("Dataset structure check")
    print("=" * 40)
    print(f"Project root: {root}")
    print(f"PlantVillage:   {pv_root}")
    print(f"PlantDoc:       {pd_root}")

    pv_layout = inspect_dataset_layout(pv_root)
    pd_layout = inspect_dataset_layout(pd_root)

    pv_classes = pv_layout.class_union() if pv_layout is not None else None
    pd_classes = pd_layout.class_union() if pd_layout is not None else None

    print("\nSummary")
    print("-------")
    if pv_layout is None:
        print("  PlantVillage: (directory unavailable)")
    else:
        kind = "direct-class" if pv_layout.structure == "direct" else "split-first"
        print(f"  PlantVillage: {kind}, {len(pv_classes or set())} class(es) in union")
    if pd_layout is None:
        print("  PlantDoc:     (directory unavailable)")
    else:
        kind = "direct-class" if pd_layout.structure == "direct" else "split-first"
        print(f"  PlantDoc:     {kind}, {len(pd_classes or set())} class(es) in union")

    if pv_layout is not None:
        _print_layout_details("PlantVillage — details", pv_layout)
    if pd_layout is not None:
        _print_layout_details("PlantDoc — details", pd_layout)

    if pv_classes is None or pd_classes is None:
        print(
            "\nNOTE: Cannot compute class overlap because one or both datasets are missing."
        )
        print("\nDone (no files were modified).")
        return

    common = pv_classes & pd_classes
    only_pv = pv_classes - pd_classes
    only_pd = pd_classes - pv_classes

    _print_class_list("Common classes (intersection, exact name match)", common)
    _print_class_list("Only in PlantVillage", only_pv)
    _print_class_list("Only in PlantDoc", only_pd)

    print("\nDone (no files were modified).")


if __name__ == "__main__":
    main()
