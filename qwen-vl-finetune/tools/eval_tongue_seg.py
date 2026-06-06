import json
import os
import sys
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

os.environ.setdefault("USE_HUB_KERNELS", "NO")
_orig_importlib_version = importlib.metadata.version


def _hide_broken_kernels_package(package_name):
    if package_name == "kernels":
        raise importlib.metadata.PackageNotFoundError(package_name)
    return _orig_importlib_version(package_name)


importlib.metadata.version = _hide_broken_kernels_package

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "qwen-vl-finetune"))

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402

from qwenvl.model import Qwen3VLSegForConditionalGeneration  # noqa: E402
from qwenvl.data.data_processor import load_seg_fields, preprocess_qwen_visual  # noqa: E402
from qwenvl.data.rope2d import get_rope_index_3  # noqa: E402


BASE_MODEL = "/home/zyzd/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct"
CHECKPOINT = "outputs/tongue_seg_phase1/model.safetensors"
ANNOTATION = "data/TongeImageDataset/val.json"
OUTPUT_DIR = "outputs/tongue_seg_eval"
MASK_SIZE = 256
MAX_OVERLAYS = 30


def arg_value(name, default):
    if name not in sys.argv:
        return default
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        raise ValueError(f"missing value for {name}")
    return sys.argv[idx + 1]


def binary_metrics(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    dice = (2 * inter + 1.0) / (pred_sum + target_sum + 1.0)
    iou = (inter + 1.0) / (union + 1.0)
    return float(dice), float(iou)


def parse_thresholds(value):
    if not value:
        return []
    return [float(item) for item in value.split(",") if item.strip()]


def make_messages(source, processor):
    data = preprocess_qwen_visual([source], processor)
    grid_thw = data["image_grid_thw"]
    position_ids, _ = get_rope_index_3(
        getattr(processor.image_processor, "merge_size", 2),
        data["input_ids"],
        image_grid_thw=grid_thw,
        video_grid_thw=None,
        second_per_grid_ts=None,
    )
    data["position_ids"] = position_ids
    return data


def overlay_image(image, gt_mask, pred_mask, box):
    image = image.convert("RGB")
    overlay = image.copy()
    gt = Image.fromarray((gt_mask * 120).astype(np.uint8), mode="L")
    pred = Image.fromarray((pred_mask * 120).astype(np.uint8), mode="L")
    gt_rgba = Image.new("RGBA", image.size, (0, 210, 90, 0))
    pred_rgba = Image.new("RGBA", image.size, (230, 40, 40, 0))
    gt_rgba.putalpha(gt)
    pred_rgba.putalpha(pred)
    overlay = Image.alpha_composite(overlay.convert("RGBA"), gt_rgba)
    overlay = Image.alpha_composite(overlay, pred_rgba).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(box, outline=(255, 210, 0), width=3)
    return overlay


def main():
    base_model_path = arg_value("--model_name_or_path", BASE_MODEL)
    checkpoint_path = Path(arg_value("--checkpoint", CHECKPOINT))
    annotation_path = Path(arg_value("--annotation", ANNOTATION))
    output_dir = Path(arg_value("--output_dir", OUTPUT_DIR))
    mask_size = int(arg_value("--seg_mask_size", str(MASK_SIZE)))
    seg_box_expand = float(arg_value("--seg_box_expand", "0.0"))
    seg_box_alpha = float(arg_value("--seg_box_alpha", "0.0"))
    seg_use_highres_fusion = arg_value("--seg_use_highres_fusion", "False").lower() == "true"
    seg_refine = arg_value("--seg_refine", "False").lower() == "true"
    strict_load = arg_value("--strict_load", "True").lower() == "true"
    threshold = float(arg_value("--threshold", "0.5"))
    threshold_sweep = parse_thresholds(arg_value("--threshold_sweep", ""))
    max_overlays = int(arg_value("--max_overlays", str(MAX_OVERLAYS)))

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(base_model_path)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        attn_implementation="sdpa",
        dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model = Qwen3VLSegForConditionalGeneration(
        base_model,
        seg_mask_size=mask_size,
        seg_box_expand=seg_box_expand,
        seg_box_alpha=seg_box_alpha,
        seg_use_highres_fusion=seg_use_highres_fusion,
        seg_refine=seg_refine,
    )
    state = torch.load(checkpoint_path, map_location="cpu") if checkpoint_path.suffix == ".pt" else None
    if state is None:
        from safetensors.torch import load_file

        state = load_file(str(checkpoint_path))
    model.load_state_dict(state, strict=strict_load)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    samples = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows = []
    sweep_rows = []
    base_path = annotation_path.parent
    for idx, sample in enumerate(samples):
        source = dict(sample)
        source["data_path"] = str(base_path)
        data = make_messages(source, processor)
        seg = load_seg_fields(source, mask_size)
        inputs = {
            "input_ids": data["input_ids"].to(device),
            "attention_mask": torch.ones_like(data["input_ids"], dtype=torch.bool).to(device),
            "position_ids": data["position_ids"].to(device),
            "pixel_values": data["pixel_values"].to(device),
            "image_grid_thw": data["image_grid_thw"].to(device),
        }
        gt_boxes = seg["gt_boxes"][None].to(device)
        seg_images = seg["seg_images"][None].to(device)
        with torch.no_grad():
            logits = model.predict_masks(
                inputs["pixel_values"],
                inputs["image_grid_thw"],
                gt_boxes,
                seg_images=seg_images,
            )
            probs = torch.sigmoid(logits)[0, 0].float().cpu()

        image = Image.open(base_path / sample["image"]).convert("RGB")
        gt_orig = (np.array(Image.open(base_path / sample["mask"]).convert("L")) > 0).astype(np.uint8)
        pred_orig = F.interpolate(
            probs[None, None],
            size=gt_orig.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        pred_bin = (pred_orig >= threshold).astype(np.uint8)
        dice, iou = binary_metrics(pred_bin, gt_orig)
        for item_threshold in threshold_sweep:
            sweep_pred = (pred_orig >= item_threshold).astype(np.uint8)
            sweep_dice, sweep_iou = binary_metrics(sweep_pred, gt_orig)
            sweep_rows.append(
                {
                    "idx": idx,
                    "image": sample["image"],
                    "threshold": item_threshold,
                    "dice": sweep_dice,
                    "miou": sweep_iou,
                    "pred_area_ratio": float(sweep_pred.mean()),
                    "gt_area_ratio": float(gt_orig.mean()),
                }
            )
        rows.append(
            {
                "idx": idx,
                "image": sample["image"],
                "mask": sample["mask"],
                "threshold": threshold,
                "dice": dice,
                "miou": iou,
                "pred_area_ratio": float(pred_bin.mean()),
                "gt_area_ratio": float(gt_orig.mean()),
                "bbox_2d": sample["bbox_2d"],
            }
        )

        if idx < max_overlays:
            overlay = overlay_image(image, gt_orig, pred_bin, sample["bbox_2d"])
            overlay.save(overlay_dir / f"{idx:03d}_{Path(sample['image']).stem}.png")

    df = pd.DataFrame(rows)
    summary = {
        "annotation": str(annotation_path),
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "samples": len(rows),
        "dice_mean": float(df["dice"].mean()),
        "dice_median": float(df["dice"].median()),
        "miou_mean": float(df["miou"].mean()),
        "miou_median": float(df["miou"].median()),
        "pred_area_ratio_mean": float(df["pred_area_ratio"].mean()),
        "gt_area_ratio_mean": float(df["gt_area_ratio"].mean()),
    }
    if sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows)
        threshold_df = (
            sweep_df.groupby("threshold", as_index=False)
            .agg(
                dice_mean=("dice", "mean"),
                dice_median=("dice", "median"),
                miou_mean=("miou", "mean"),
                miou_median=("miou", "median"),
                pred_area_ratio_mean=("pred_area_ratio", "mean"),
                gt_area_ratio_mean=("gt_area_ratio", "mean"),
            )
            .sort_values(["dice_mean", "miou_mean"], ascending=False)
        )
        best = threshold_df.iloc[0].to_dict()
        summary["best_threshold"] = float(best["threshold"])
        summary["best_threshold_dice_mean"] = float(best["dice_mean"])
        summary["best_threshold_miou_mean"] = float(best["miou_mean"])
        sweep_df.to_json(output_dir / "threshold_predictions.jsonl", orient="records", lines=True, force_ascii=False)
        threshold_df.to_excel(output_dir / "threshold_sweep.xlsx", index=False)
        (output_dir / "threshold_sweep.json").write_text(
            json.dumps(threshold_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_json(output_dir / "predictions.jsonl", orient="records", lines=True, force_ascii=False)
    df.to_excel(output_dir / "metrics.xlsx", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
