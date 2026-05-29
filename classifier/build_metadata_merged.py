import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan ISIC class folders and build metadata_merged.csv."
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="metadata_merged.csv")
    parser.add_argument(
        "--exts",
        type=str,
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
    )
    parser.add_argument(
        "--allow_missing_classes",
        action="store_true",
        help="If set, missing class folders are allowed. Otherwise missing folders raise an error.",
    )
    return parser.parse_args()


def normalize_exts(exts):
    normalized = set()
    for ext in exts:
        ext = ext.lower()
        if not ext.startswith("."):
            ext = "." + ext
        normalized.add(ext)
    return normalized


def main():
    args = parse_args()

    root = Path(args.root).resolve()
    output_csv = root / args.output_name
    exts = normalize_exts(args.exts)

    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    missing_classes = []
    rows = []
    seen_paths = set()

    print("=" * 100)
    print(f"[INFO] root       : {root}")
    print(f"[INFO] output csv : {output_csv}")
    print(f"[INFO] extensions : {sorted(exts)}")
    print("=" * 100)

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name

        print("-" * 100)
        print(f"[CLASS] {class_name}")
        print(f"[PATH ] {class_dir}")

        if not class_dir.exists():
            missing_classes.append(class_name)
            print(f"[WARN] missing class folder: {class_dir}")
            continue

        if not class_dir.is_dir():
            raise NotADirectoryError(
                f"Class path exists but is not a directory: {class_dir}"
            )

        all_files = [p for p in class_dir.rglob("*") if p.is_file()]

        suffix_counter = Counter(
            p.suffix.lower() if p.suffix else "<no_suffix>" for p in all_files
        )

        print(f"[DEBUG] total files under {class_name}: {len(all_files)}")
        print(f"[DEBUG] suffix distribution: {dict(suffix_counter)}")

        image_paths = []

        for img_path in sorted(all_files):
            if img_path.suffix.lower() not in exts:
                continue

            abs_path = img_path.resolve()
            abs_path_str = str(abs_path)

            if abs_path_str in seen_paths:
                print(f"[WARN] duplicated path skipped: {abs_path}")
                continue

            seen_paths.add(abs_path_str)
            image_paths.append(abs_path)

        print(f"[INFO] valid image paths found in {class_name}: {len(image_paths)}")

        for abs_path in image_paths:
            rel_path = abs_path.relative_to(root)

            rows.append(
                {
                    "img_path": str(rel_path),
                    "abs_path": str(abs_path),
                    "label": class_name,
                    "label_idx": label_idx,
                    "image_id": str(rel_path).replace("\\", "/").replace("/", "__"),
                    "source_root": str(root),
                }
            )

    if missing_classes and not args.allow_missing_classes:
        raise FileNotFoundError(
            f"Missing class folders: {missing_classes}. "
            f"Use --allow_missing_classes only if this is intentional."
        )

    if len(rows) == 0:
        raise RuntimeError(
            f"No images found under root: {root}. "
            f"Expected class folders: {CLASS_NAMES}"
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["label_idx", "img_path"]).reset_index(drop=True)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    with open(root / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("[INFO] class distribution:")
    print(df.groupby("label").size().reindex(CLASS_NAMES, fill_value=0))
    print(f"\n[INFO] total images: {len(df)}")
    print(f"[INFO] saved to: {output_csv}")
    print(f"[INFO] label map saved to: {root / 'label_map.json'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
