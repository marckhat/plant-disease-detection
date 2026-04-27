"""Build aligned classification datasets from PlantVillage and PlantDoc using a fixed class mapping."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

# PlantDoc folder name -> PlantVillage folder name (use EXACTLY as given).
PLANTDOC_TO_PLANTVILLAGE: dict[str, str] = {
    "Apple_Scab_Leaf": "Apple___Apple_scab",
    "Apple_leaf": "Apple___healthy",
    "Apple_rust_leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper_leaf": "Pepper,_bell___healthy",
    "Bell_pepper_leaf_spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry_leaf": "Blueberry___healthy",
    "Cherry_leaf": "Cherry_(including_sour)___healthy",
    "Corn_Gray_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_leaf_blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_rust_leaf": "Corn_(maize)___Common_rust_",
    "Peach_leaf": "Peach___healthy",
    "Potato_leaf_early_blight": "Potato___Early_blight",
    "Potato_leaf_late_blight": "Potato___Late_blight",
    "Raspberry_leaf": "Raspberry___healthy",
    "Soyabean_leaf": "Soybean___healthy",
    "Squash_Powdery_mildew_leaf": "Squash___Powdery_mildew",
    "Strawberry_leaf": "Strawberry___healthy",
    "Tomato_leaf": "Tomato___healthy",
    "Tomato_Early_blight_leaf": "Tomato___Early_blight",
    "Tomato_Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato_leaf_bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato_leaf_late_blight": "Tomato___Late_blight",
    "Tomato_leaf_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Tomato_leaf_yellow_virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mold_leaf": "Tomato___Leaf_Mold",
    "grape_leaf": "Grape___healthy",
    "grape_leaf_black_rot": "Grape___Black_rot",
}

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})

PLANTVILLAGE_SPLITS = ("train", "val")
PLANTDOC_SPLITS = ("train", "test")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def unified_label(plantvillage_folder_name: str) -> str:
    """
    Unified folder label: crop + condition together.

    PlantVillage releases use ``___`` between crop and class; we normalize to ``__``
    for output folders, e.g. ``Tomato___Early_blight`` -> ``Tomato__Early_blight``.
    """
    return plantvillage_folder_name.replace("___", "__")


def list_images_in_class_dir(class_dir: Path) -> list[Path]:
    """All image files under ``class_dir`` (recursive), with supported extensions."""
    if not class_dir.is_dir():
        return []
    out: list[Path] = []
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return sorted(out)


def unique_destination(dest_path: Path) -> Path:
    """If ``dest_path`` exists, append _1, _2, ... before the suffix."""
    if not dest_path.exists():
        return dest_path
    stem = dest_path.stem
    suf = dest_path.suffix
    parent = dest_path.parent
    k = 1
    while True:
        cand = parent / f"{stem}_{k}{suf}"
        if not cand.exists():
            return cand
        k += 1


def class_folder_has_images(raw_root: Path, split: str, folder_name: str) -> bool:
    """True if ``raw_root/split/folder_name`` exists and contains at least one image."""
    src_dir = raw_root / split / folder_name
    if not src_dir.is_dir():
        return False
    return bool(list_images_in_class_dir(src_dir))


def mapping_present_in_plantvillage(raw_pv: Path, pv_folder_name: str) -> bool:
    return any(class_folder_has_images(raw_pv, sp, pv_folder_name) for sp in PLANTVILLAGE_SPLITS)


def mapping_present_in_plantdoc(raw_pd: Path, pd_folder_name: str) -> bool:
    return any(class_folder_has_images(raw_pd, sp, pd_folder_name) for sp in PLANTDOC_SPLITS)


def copy_images_to_class_dir(
    src_files: list[Path],
    dst_class_dir: Path,
) -> int:
    dst_class_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in src_files:
        dst = unique_destination(dst_class_dir / src.name)
        shutil.copy2(src, dst)
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build aligned datasets under data/processed/aligned/ from raw splits."
    )
    # Default: no filtering. Per-split min-count rules are easy to misuse: the same
    # mapped class can be kept in one split but dropped in another, producing an
    # inconsistent label space across splits. Prefer fixing data or filtering globally
    # after a full inventory instead of enabling this unless you know you need it.
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        metavar="N",
        help="(Optional) Skip a source class folder with fewer than N images per split. "
        "Default: no filtering.",
    )
    args = parser.parse_args()
    min_samples: int | None = args.min_samples

    root = project_root()
    raw_pv = root / "data" / "raw" / "plantvillage"
    raw_pd = root / "data" / "raw" / "plantdoc"
    out_base = root / "data" / "processed" / "aligned"

    # Per unified label -> image counts for each output dataset
    counts_pv: dict[str, int] = defaultdict(int)
    counts_pd: dict[str, int] = defaultdict(int)
    total_copied = 0

    # PlantVillage: copy from train/val using PlantVillage folder names (mapping values)
    for split in PLANTVILLAGE_SPLITS:
        split_out = out_base / "plantvillage" / split
        seen_pv_names = set(PLANTDOC_TO_PLANTVILLAGE.values())
        for pv_folder_name in sorted(seen_pv_names):
            unified = unified_label(pv_folder_name)
            src_dir = raw_pv / split / pv_folder_name
            if not src_dir.is_dir():
                print(f"WARNING: missing folder (skipped): {src_dir}")
                continue
            files = list_images_in_class_dir(src_dir)
            if min_samples is not None and len(files) < min_samples:
                print(
                    f"SKIP PlantVillage/{split}/{pv_folder_name!r}: "
                    f"{len(files)} images (< {min_samples})"
                )
                continue
            if not files:
                print(f"WARNING: no images in {src_dir}")
                continue
            dst_dir = split_out / unified
            n = copy_images_to_class_dir(files, dst_dir)
            counts_pv[unified] += n
            total_copied += n

    # PlantDoc: copy from train/test using PlantDoc folder names (mapping keys)
    for split in PLANTDOC_SPLITS:
        split_out = out_base / "plantdoc" / split
        for pd_folder_name, pv_folder_name in sorted(PLANTDOC_TO_PLANTVILLAGE.items()):
            unified = unified_label(pv_folder_name)
            src_dir = raw_pd / split / pd_folder_name
            if not src_dir.is_dir():
                print(f"WARNING: missing folder (skipped): {src_dir}")
                continue
            files = list_images_in_class_dir(src_dir)
            if min_samples is not None and len(files) < min_samples:
                print(
                    f"SKIP PlantDoc/{split}/{pd_folder_name!r}: "
                    f"{len(files)} images (< {min_samples})"
                )
                continue
            if not files:
                print(f"WARNING: no images in {src_dir}")
                continue
            dst_dir = split_out / unified
            n = copy_images_to_class_dir(files, dst_dir)
            counts_pd[unified] += n
            total_copied += n

    labels_union = sorted(set(counts_pv) | set(counts_pd))

    found_both: list[tuple[str, str, str]] = []
    missing_pv: list[tuple[str, str, str]] = []
    missing_pd: list[tuple[str, str, str]] = []
    for pd_key, pv_val in sorted(PLANTDOC_TO_PLANTVILLAGE.items()):
        u = unified_label(pv_val)
        in_pv = mapping_present_in_plantvillage(raw_pv, pv_val)
        in_pd = mapping_present_in_plantdoc(raw_pd, pd_key)
        row = (u, pd_key, pv_val)
        if in_pv and in_pd:
            found_both.append(row)
        else:
            if not in_pv:
                missing_pv.append(row)
            if not in_pd:
                missing_pd.append(row)

    print("\n" + "=" * 50)
    print("Aligned dataset build — summary")
    print("=" * 50)
    print(f"Output root: {out_base}")
    if min_samples is not None:
        print(f"Min samples filter: {min_samples} (per source class folder, per split)")
    else:
        print("Min samples filter: off (default)")

    print(
        f"\nMapped classes present in BOTH raw datasets ({len(found_both)} "
        f"of {len(PLANTDOC_TO_PLANTVILLAGE)}):"
    )
    if not found_both:
        print("  (none)")
    else:
        for u, pd_key, pv_val in found_both:
            print(f"  {u}  |  PlantDoc: {pd_key!r}  ->  PlantVillage: {pv_val!r}")

    print(f"\nMapped classes missing from PlantVillage ({len(missing_pv)}):")
    if not missing_pv:
        print("  (none)")
    else:
        for u, pd_key, pv_val in missing_pv:
            print(f"  {u}  |  PlantDoc: {pd_key!r}  ->  PlantVillage: {pv_val!r}")

    print(f"\nMapped classes missing from PlantDoc ({len(missing_pd)}):")
    if not missing_pd:
        print("  (none)")
    else:
        for u, pd_key, pv_val in missing_pd:
            print(f"  {u}  |  PlantDoc: {pd_key!r}  ->  PlantVillage: {pv_val!r}")

    print(f"\nUnified labels with at least one copied image: {len(labels_union)}")
    print("\nImages per unified class (PlantVillage, all splits combined):")
    for u in sorted(counts_pv):
        print(f"  {u}: {counts_pv[u]}")
    if not counts_pv:
        print("  (none)")

    print("\nImages per unified class (PlantDoc, all splits combined):")
    if not counts_pd:
        print("  (none)")
    else:
        for u in sorted(counts_pd):
            print(f"  {u}: {counts_pd[u]}")

    print(f"\nTotal images copied: {total_copied}")
    print("\nDone.")


if __name__ == "__main__":
    main()
