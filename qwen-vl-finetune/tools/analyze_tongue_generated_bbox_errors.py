import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("data/TongeImageDataset")
OUTPUT = Path("outputs/tongue_seg_p1_prompt_summary/bbox_error_analysis.json")
INPUTS = [
    ("val", Path("outputs/tongue_seg_phase2b_eval_val_generated/predictions.jsonl")),
    ("test", Path("outputs/tongue_seg_phase2b_eval_test_generated/predictions.jsonl")),
]


def parse_generated_bbox(text):
    match = re.search(r"""["']bbox_2d["']\s*:\s*\[([^\]]+)\]""", text)
    if not match:
        return None
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))]
    return nums if len(nums) == 4 else None


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    return [
        min(max(x1, 0.0), width),
        min(max(y1, 0.0), height),
        min(max(x2, 0.0), width),
        min(max(y2, 0.0), height),
    ]


def center(box):
    return [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]


def wh(box):
    return [max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])]


def summarize(rows):
    if not rows:
        return {}
    keys = [
        "old_bbox_iou",
        "qwen1000_bbox_iou",
        "center_dx",
        "center_dy",
        "center_distance",
        "width_ratio",
        "height_ratio",
        "area_ratio",
    ]
    out = {"count": len(rows)}
    for key in keys:
        vals = np.array([row[key] for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_median"] = float(np.median(vals))
        out[f"{key}_min"] = float(vals.min())
        out[f"{key}_max"] = float(vals.max())
    out["old_right_clipped"] = int(sum(row["old_right_clipped"] for row in rows))
    out["old_bottom_clipped"] = int(sum(row["old_bottom_clipped"] for row in rows))
    out["right_down_shift_count"] = int(sum(row["center_dx"] > 0 and row["center_dy"] > 0 for row in rows))
    return out


def main():
    all_rows = []
    by_split = {}
    parse_success = {}
    for split, path in INPUTS:
        rows = []
        total = 0
        success = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            total += 1
            raw = parse_generated_bbox(item.get("generated_text") or "")
            if raw is None:
                continue
            success += 1
            image = Image.open(ROOT / item["image"])
            width, height = image.size
            gt = [float(x) for x in item["bbox_2d"]]
            old = [float(x) for x in item["eval_bbox_2d"]]
            qwen = clamp_box(
                [raw[0] / 1000 * width, raw[1] / 1000 * height, raw[2] / 1000 * width, raw[3] / 1000 * height],
                width,
                height,
            )
            gt_c = center(gt)
            old_c = center(old)
            gt_w, gt_h = wh(gt)
            old_w, old_h = wh(old)
            row = {
                "split": split,
                "idx": item["idx"],
                "image": item["image"],
                "generated_text": item.get("generated_text"),
                "gt_bbox": item["bbox_2d"],
                "old_eval_bbox": item["eval_bbox_2d"],
                "qwen1000_eval_bbox": [round(x, 3) for x in qwen],
                "old_bbox_iou": box_iou(old, gt),
                "qwen1000_bbox_iou": box_iou(qwen, gt),
                "center_dx": old_c[0] - gt_c[0],
                "center_dy": old_c[1] - gt_c[1],
                "center_distance": float(np.hypot(old_c[0] - gt_c[0], old_c[1] - gt_c[1])),
                "width_ratio": old_w / gt_w if gt_w > 0 else 0.0,
                "height_ratio": old_h / gt_h if gt_h > 0 else 0.0,
                "area_ratio": (old_w * old_h) / (gt_w * gt_h) if gt_w > 0 and gt_h > 0 else 0.0,
                "old_right_clipped": old[2] >= width,
                "old_bottom_clipped": old[3] >= height,
            }
            rows.append(row)
            all_rows.append(row)
        parse_success[split] = {"success": success, "total": total, "rate": success / total if total else 0.0}
        by_split[split] = summarize(rows)

    output = {
        "inputs": [str(path) for _, path in INPUTS],
        "parse_success": parse_success,
        "summary": {**by_split, "combined": summarize(all_rows)},
        "worst_old_bbox_iou": sorted(all_rows, key=lambda row: row["old_bbox_iou"])[:5],
        "conclusion": "Generated bbox values are consistent with Qwen-style 0-1000 coordinates; old pixel interpretation caused systematic right/down shift and clipping.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"]["combined"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
