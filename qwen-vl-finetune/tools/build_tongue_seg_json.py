import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("data/TongeImageDataset")
PROMPT = "<image>\nLocate and segment the tongue, report bbox coordinates and mask in JSON format."


def mask_to_bbox(mask_path):
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError(f"empty mask: {mask_path}")
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def make_item(image_path):
    mask_path = ROOT / "origin_GT" / image_path.name
    if not mask_path.exists():
        raise FileNotFoundError(f"missing mask for {image_path}: {mask_path}")

    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.size != mask.size:
            raise ValueError(f"size mismatch: {image_path} {image.size} vs {mask_path} {mask.size}")

    bbox = mask_to_bbox(mask_path)
    answer = json.dumps(
        [{"bbox_2d": bbox, "label": "tongue", "mask": "<mask_start><mask_token><mask_end>"}],
        ensure_ascii=False,
    )
    return {
        "image": f"origin_Image/{image_path.name}",
        "mask": f"origin_GT/{mask_path.name}",
        "bbox_2d": bbox,
        "label": "tongue",
        "conversations": [
            {"from": "human", "value": PROMPT},
            {"from": "gpt", "value": answer},
        ],
    }


def write_json(path, items):
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(items)} samples to {path}")


def main():
    image_dir = ROOT / "origin_Image"
    images = sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem))
    if len(images) != 300:
        raise ValueError(f"expected 300 images, got {len(images)}")

    items = [make_item(path) for path in images]
    write_json(ROOT / "train.json", items[:240])
    write_json(ROOT / "val.json", items[240:270])
    write_json(ROOT / "test.json", items[270:])


if __name__ == "__main__":
    main()
