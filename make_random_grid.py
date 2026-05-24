import argparse
import os
import random

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample 16 training images and save a 4x4 grid."
    )

    parser.add_argument(
        "--gt-csv",
        type=str,
        default="dataset/ISIC2018_Task3_Training_GroundTruth.csv",
        help="训练集 GroundTruth CSV 路径。",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="dataset/ISIC2018_Task3_Training_Input",
        help="训练集图片目录。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="random_train_4x4_grid.jpg",
        help="输出网格图片路径。",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="每张小图缩放后的尺寸。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--show-label",
        action="store_true",
        help="是否在每张图左上角显示类别名。",
    )

    return parser.parse_args()


def get_label_from_row(row, class_columns):
    """从 one-hot 标签行中取类别名。"""
    label_idx = row[class_columns].values.astype(float).argmax()
    return class_columns[label_idx]


def make_grid(images, labels, image_ids, image_size, show_label=False):
    grid_rows = 4
    grid_cols = 4
    canvas_w = grid_cols * image_size
    canvas_h = grid_rows * image_size

    grid = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols

        x = col * image_size
        y = row * image_size

        img = img.resize((image_size, image_size), Image.BICUBIC)
        grid.paste(img, (x, y))

        if show_label:
            text = f"{labels[idx]} | {image_ids[idx]}"
            draw.rectangle((x, y, x + image_size, y + 22), fill=(0, 0, 0))
            draw.text((x + 4, y + 4), text, fill=(255, 255, 255))

    return grid


def main():
    args = parse_args()

    random.seed(args.seed)

    df = pd.read_csv(args.gt_csv)

    if "image" not in df.columns:
        raise ValueError("CSV 中必须包含 image 列。")

    class_columns = [c for c in df.columns if c != "image"]

    if len(df) < 16:
        raise ValueError(f"训练集样本不足 16 张，当前只有 {len(df)} 张。")

    sampled_df = df.sample(n=16, random_state=args.seed).reset_index(drop=True)

    images = []
    labels = []
    image_ids = []

    for _, row in sampled_df.iterrows():
        image_id = str(row["image"])
        img_path = os.path.join(args.img_dir, f"{image_id}.jpg")

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"找不到图片文件: {img_path}")

        img = Image.open(img_path).convert("RGB")
        label = get_label_from_row(row, class_columns)

        images.append(img)
        labels.append(label)
        image_ids.append(image_id)

    grid = make_grid(
        images=images,
        labels=labels,
        image_ids=image_ids,
        image_size=args.image_size,
        show_label=args.show_label,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    grid.save(args.output, quality=95)

    print(f"[DONE] saved 4x4 grid to: {args.output}")


if __name__ == "__main__":
    main()
