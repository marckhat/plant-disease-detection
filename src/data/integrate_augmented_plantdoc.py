"""Copy augmented_plantdoc into data/processed/aligned/ with unified class labels."""

from __future__ import annotations

import shutil
from pathlib import Path

# Maps augmented_plantdoc folder name -> unified label (same convention as aligned datasets).
AUGMENTED_TO_UNIFIED: dict[str, str] = {
    "Apple leaf":                  "Apple__healthy",
    "Apple rust leaf":             "Apple__Cedar_apple_rust",
    "Apple Scab Leaf":             "Apple__Apple_scab",
    "Bell_pepper leaf":            "Pepper,_bell__healthy",
    "Bell_pepper leaf spot":       "Pepper,_bell__Bacterial_spot",
    "Blueberry leaf":              "Blueberry__healthy",
    "Cherry leaf":                 "Cherry_(including_sour)__healthy",
    "Corn Gray leaf spot":         "Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight":            "Corn_(maize)__Northern_Leaf_Blight",
    "Corn rust leaf":              "Corn_(maize)__Common_rust_",
    "grape leaf":                  "Grape__healthy",
    "grape leaf black rot":        "Grape__Black_rot",
    "Peach leaf":                  "Peach__healthy",
    "Potato leaf early blight":    "Potato__Early_blight",
    "Potato leaf late blight":     "Potato__Late_blight",
    "Raspberry leaf":              "Raspberry__healthy",
    "Soyabean leaf":               "Soybean__healthy",
    "Squash Powdery mildew leaf":  "Squash__Powdery_mildew",
    "Strawberry leaf":             "Strawberry__healthy",
    "Tomato Early blight leaf":    "Tomato__Early_blight",
    "Tomato leaf":                 "Tomato__healthy",
    "Tomato leaf bacterial spot":  "Tomato__Bacterial_spot",
    "Tomato leaf late blight":     "Tomato__Late_blight",
    "Tomato leaf mosaic virus":    "Tomato__Tomato_mosaic_virus",
    "Tomato leaf yellow virus":    "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf":            "Tomato__Leaf_Mold",
    "Tomato Septoria leaf spot":   "Tomato__Septoria_leaf_spot",
}

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
SPLITS = ("train", "val", "test")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def unique_dst(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suf, parent = path.stem, path.suffix, path.parent
    k = 1
    while True:
        cand = parent / f"{stem}_{k}{suf}"
        if not cand.exists():
            return cand
        k += 1


def main() -> None:
    src_root = Path.home() / "archive" / "augmented_plantdoc"
    dst_root = project_root() / "data" / "processed" / "aligned" / "augmented_plantdoc"

    if not src_root.exists():
        raise FileNotFoundError(f"Source not found: {src_root}")

    total_copied = 0
    counts: dict[str, dict[str, int]] = {sp: {} for sp in SPLITS}
    missing_src: list[str] = []

    for split in SPLITS:
        split_src = src_root / split
        if not split_src.is_dir():
            print(f"WARNING: split folder missing: {split_src}")
            continue

        for aug_name, unified in sorted(AUGMENTED_TO_UNIFIED.items()):
            src_class = split_src / aug_name
            if not src_class.is_dir():
                missing_src.append(f"{split}/{aug_name}")
                continue

            images = list_images(src_class)
            if not images:
                print(f"WARNING: no images in {src_class}")
                continue

            dst_class = dst_root / split / unified
            dst_class.mkdir(parents=True, exist_ok=True)

            for img in images:
                dst = unique_dst(dst_class / img.name)
                shutil.copy2(img, dst)
                total_copied += 1

            counts[split][unified] = len(images)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("augmented_plantdoc integration — summary")
    print("=" * 55)
    print(f"Source:      {src_root}")
    print(f"Destination: {dst_root}")
    print()

    for split in SPLITS:
        if not counts[split]:
            continue
        n = sum(counts[split].values())
        print(f"[{split}]  {len(counts[split])} classes  |  {n} images")
        for unified in sorted(counts[split]):
            print(f"    {unified}: {counts[split][unified]}")

    if missing_src:
        print(f"\nMissing source folders ({len(missing_src)}):")
        for m in missing_src:
            print(f"  {m}")

    print(f"\nTotal images copied: {total_copied}")
    print("Done.")


if __name__ == "__main__":
    main()
