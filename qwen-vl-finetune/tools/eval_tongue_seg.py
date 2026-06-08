import json
import os
import re
import sys
import hashlib
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
PROMPT = "<image>\nLocate and segment the tongue, report bbox coordinates and mask in JSON format."


def arg_value(name, default):
    if name not in sys.argv:
        return default
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        raise ValueError(f"missing value for {name}")
    return sys.argv[idx + 1]


def bool_arg(name, default):
    return str(arg_value(name, str(default))).lower() == "true"


def load_run_config(checkpoint_path):
    config_path = checkpoint_path.parent / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def binary_metrics(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    pred_sum = int(pred.sum())
    target_sum = int(target.sum())
    dice = (2 * inter + 1.0) / (pred_sum + target_sum + 1.0)
    miou = (inter + 1.0) / (union + 1.0)
    iou = inter / union if union > 0 else 1.0
    return float(dice), float(miou), float(iou), inter, union, pred_sum, target_sum


def parse_box(value):
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", value)]
    if len(nums) != 4:
        raise ValueError(f"expected 4 bbox numbers, got: {value}")
    return nums


def parse_generated_bbox(text):
    match = re.search(r"""["']bbox_2d["']\s*:\s*\[([^\]]+)\]""", text)
    if not match:
        raise ValueError(f"could not find bbox_2d in generated text: {text}")
    return parse_box(match.group(1))


def normalize_box(box, width, height, coord_format="auto"):
    if coord_format == "auto":
        coord_format = "normalized" if max(box) <= 1.0 else "pixel"
    if coord_format == "normalized":
        norm = torch.tensor(box, dtype=torch.float32)
    elif coord_format == "pixel":
        norm = torch.tensor(
            [box[0] / width, box[1] / height, box[2] / width, box[3] / height],
            dtype=torch.float32,
        )
    elif coord_format == "qwen1000":
        norm = torch.tensor([box[0] / 1000, box[1] / 1000, box[2] / 1000, box[3] / 1000], dtype=torch.float32)
    else:
        raise ValueError("--bbox_coord_format must be auto, pixel, normalized, or qwen1000")
    norm = norm.clamp(0, 1)
    if norm[2] <= norm[0] or norm[3] <= norm[1]:
        raise ValueError(f"invalid bbox after normalization: {box}")
    return norm


def box_to_pixels(box, width, height):
    if max(box) <= 1.0:
        x1, y1, x2, y2 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    else:
        x1, y1, x2, y2 = box
    x1 = min(max(int(round(x1)), 0), width)
    y1 = min(max(int(round(y1)), 0), height)
    x2 = min(max(int(round(x2)), 0), width)
    y2 = min(max(int(round(y2)), 0), height)
    return [x1, y1, x2, y2]


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(x) for x in box_a]
    bx1, by1, bx2, by2 = [float(x) for x in box_b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def jitter_box(box, amount, rng):
    if amount <= 0:
        return box.clone()
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) * 0.5 + rng.uniform(-amount, amount) * w
    cy = (y1 + y2) * 0.5 + rng.uniform(-amount, amount) * h
    w = max(1e-4, w * (1.0 + rng.uniform(-amount, amount)))
    h = max(1e-4, h * (1.0 + rng.uniform(-amount, amount)))
    out = torch.tensor(
        [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5],
        dtype=torch.float32,
    ).clamp(0, 1)
    if out[2] <= out[0] or out[3] <= out[1]:
        return box.clone()
    return out


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


def prompt_text(prompt):
    return prompt.replace("<image>\n", "").replace("<image>", "").strip()


def load_prompt(prompt_file, prompt):
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    return prompt


def hash_prompt(prompt):
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]


def make_generation_inputs(image, processor, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text(prompt)},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def generate_bbox(model, processor, image, device, max_new_tokens, prompt):
    inputs = to_device(make_generation_inputs(image, processor, prompt), device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output_ids = model.base_model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    text = processor.tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)
    return text, inputs


def mask_rgba(mask, color, alpha):
    rgba = Image.new("RGBA", mask.shape[::-1], (*color, 0))
    rgba.putalpha(Image.fromarray((mask.astype(np.uint8) * alpha), mode="L"))
    return rgba


def overlay_image(image, gt_mask, pred_mask, box, alpha):
    image = image.convert("RGB")
    overlay = Image.alpha_composite(image.convert("RGBA"), mask_rgba(gt_mask, (0, 210, 90), alpha))
    overlay = Image.alpha_composite(overlay, mask_rgba(pred_mask, (230, 40, 40), alpha)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(box, outline=(255, 210, 0), width=3)
    return overlay


def single_mask_overlay(image, mask, color, alpha, box=None):
    image = image.convert("RGB")
    overlay = Image.alpha_composite(image.convert("RGBA"), mask_rgba(mask, color, alpha)).convert("RGB")
    if box is not None:
        ImageDraw.Draw(overlay).rectangle(box, outline=(255, 210, 0), width=3)
    return overlay


def error_overlay(image, gt_mask, pred_mask, alpha, box):
    image = image.convert("RGB")
    tp = np.logical_and(gt_mask, pred_mask)
    fp = np.logical_and(~gt_mask.astype(bool), pred_mask.astype(bool))
    fn = np.logical_and(gt_mask.astype(bool), ~pred_mask.astype(bool))
    overlay = Image.alpha_composite(image.convert("RGBA"), mask_rgba(tp, (0, 210, 90), alpha))
    overlay = Image.alpha_composite(overlay, mask_rgba(fp, (230, 40, 40), alpha))
    overlay = Image.alpha_composite(overlay, mask_rgba(fn, (40, 110, 255), alpha)).convert("RGB")
    ImageDraw.Draw(overlay).rectangle(box, outline=(255, 210, 0), width=3)
    return overlay


def add_title(image, title):
    pad = 24
    canvas = Image.new("RGB", (image.width, image.height + pad), "white")
    canvas.paste(image, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(20, 20, 20))
    return canvas


def resize_panel(image, width):
    height = int(image.height * width / image.width)
    return image.resize((width, height), Image.BILINEAR)


def save_overview(vis_items, output_path, alpha, count):
    if count <= 0 or not vis_items:
        return
    items = sorted(vis_items, key=lambda item: item["dice"])[:count]
    panel_w = 240
    rows = []
    for item in items:
        image = item["image"]
        gt = item["gt_mask"]
        pred = item["pred_mask"]
        box = item["box"]
        panels = [
            add_title(resize_panel(image, panel_w), f"{item['idx']:03d} original"),
            add_title(resize_panel(single_mask_overlay(image, gt, (0, 210, 90), alpha, box), panel_w), "GT"),
            add_title(resize_panel(single_mask_overlay(image, pred, (230, 40, 40), alpha, box), panel_w), "Prediction"),
            add_title(resize_panel(error_overlay(image, gt, pred, alpha, box), panel_w), f"Error Dice {item['dice']:.3f}"),
        ]
        row = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        rows.append(row)
    gap = 10
    canvas = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows) + gap * (len(rows) - 1)), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + gap
    canvas.save(output_path)


def write_failure_cases(rows, output_path, count):
    if count <= 0 or not rows:
        return
    cases = []
    valid_rows = [
        row
        for row in rows
        if row.get("bbox_parse_success", True) and row.get("dice") is not None and not pd.isna(row["dice"])
    ]
    for row in sorted(valid_rows, key=lambda item: item["dice"])[:count]:
        delta = row["pred_area_ratio"] - row["gt_area_ratio"]
        failure_modes = []
        if delta < -0.015:
            failure_modes.extend(["tongue_area_too_small", "tongue_tip_or_edge_missing"])
            note = "Prediction area is smaller than GT; inspect overlay for missed tongue edge or tip."
        elif delta > 0.015:
            failure_modes.extend(["tongue_area_too_large", "tongue_boundary_expansion"])
            note = "Prediction area is larger than GT; inspect overlay for mild boundary expansion."
        else:
            failure_modes.append("local_boundary_error")
            note = "Area ratio is close to GT; remaining error is mostly local boundary mismatch."
        if row["dice"] < 0.96:
            failure_modes.append("manual_review_for_lip_teeth_background_or_lighting")
        cases.append(
            {
                "idx": row["idx"],
                "image": row["image"],
                "mask": row["mask"],
                "dice": row["dice"],
                "miou": row["miou"],
                "pred_area_ratio": row["pred_area_ratio"],
                "gt_area_ratio": row["gt_area_ratio"],
                "area_delta": delta,
                "bbox_2d": row["bbox_2d"],
                "eval_bbox_2d": row.get("eval_bbox_2d"),
                "bbox_iou": row.get("bbox_iou"),
                "failure_modes": failure_modes,
                "note": note,
            }
        )
    summary = {
        "selection": f"{len(cases)} lowest Dice samples",
        "review_categories": [
            "tongue_tip_or_edge_missing",
            "tongue_boundary_expansion",
            "lip_teeth_or_background_false_positive",
            "lighting_abnormal",
            "tongue_area_too_small",
            "tongue_area_too_large",
        ],
        "main_observation": "Residual errors are dominated by local tongue boundary mismatch and mild under-segmentation; inspect overlays for lip, teeth, background, or lighting failures.",
        "cases": cases,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    checkpoint_path = Path(arg_value("--checkpoint", CHECKPOINT))
    config = load_run_config(checkpoint_path)
    base_model_path = arg_value("--model_name_or_path", config.get("model_name_or_path", BASE_MODEL))
    annotation_path = Path(arg_value("--annotation", ANNOTATION))
    output_dir = Path(arg_value("--output_dir", OUTPUT_DIR))
    mask_size = int(arg_value("--seg_mask_size", config.get("seg_mask_size", MASK_SIZE)))
    seg_box_expand = float(arg_value("--seg_box_expand", config.get("seg_box_expand", 0.0)))
    seg_box_alpha = float(arg_value("--seg_box_alpha", config.get("seg_box_alpha", 0.0)))
    seg_use_highres_fusion = bool_arg("--seg_use_highres_fusion", config.get("seg_use_highres_fusion", False))
    seg_refine = bool_arg("--seg_refine", config.get("seg_refine", False))
    seg_use_box_film = bool_arg("--seg_use_box_film", config.get("seg_use_box_film", False))
    seg_box_fourier_bands = int(arg_value("--seg_box_fourier_bands", config.get("seg_box_fourier_bands", 8)))
    strict_load = arg_value("--strict_load", "True").lower() == "true"
    threshold = float(arg_value("--threshold", "0.5"))
    threshold_sweep = parse_thresholds(arg_value("--threshold_sweep", ""))
    bbox_source = arg_value("--bbox_source", "gt")
    bbox_jitter = float(arg_value("--bbox_jitter", "0.0"))
    bbox_jitter_seed = int(arg_value("--bbox_jitter_seed", "0"))
    bbox_coord_format = arg_value("--bbox_coord_format", "qwen1000" if bbox_source == "generated" else "auto")
    prompt_file = arg_value("--prompt_file", "")
    prompt = load_prompt(prompt_file, arg_value("--prompt", PROMPT))
    prompt_name = arg_value(
        "--prompt_name",
        Path(prompt_file).stem if prompt_file else ("default" if prompt == PROMPT else "custom"),
    )
    prompt_hash = hash_prompt(prompt)
    max_new_tokens = int(arg_value("--max_new_tokens", "128"))
    max_overlays = int(arg_value("--max_overlays", str(MAX_OVERLAYS)))
    overlay_alpha = int(arg_value("--overlay_alpha", "90"))
    overlay_top_k_worst = int(arg_value("--overlay_top_k_worst", "0"))
    overview_count = int(arg_value("--overview_count", "0"))
    failure_cases_count = int(arg_value("--failure_cases_count", "10"))
    if bbox_source not in {"gt", "generated"}:
        raise ValueError("--bbox_source must be gt or generated")
    if bbox_source == "generated" and bbox_jitter > 0:
        raise ValueError("--bbox_jitter is only supported with --bbox_source gt")

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for old_overlay in overlay_dir.glob("*.png"):
        old_overlay.unlink()

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
        seg_use_box_film=seg_use_box_film,
        seg_box_fourier_bands=seg_box_fourier_bands,
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
    vis_items = []
    base_path = annotation_path.parent
    rng = np.random.default_rng(bbox_jitter_seed)
    for idx, sample in enumerate(samples):
        source = dict(sample)
        source["data_path"] = str(base_path)
        seg = load_seg_fields(source, mask_size)
        seg_images = seg["seg_images"][None].to(device)
        image = Image.open(base_path / sample["image"]).convert("RGB")
        image.load()
        gt_orig = (np.array(Image.open(base_path / sample["mask"]).convert("L")) > 0).astype(np.uint8)
        gt_box = seg["gt_boxes"]
        generated_text = None
        bbox_parse_success = True
        bbox_parse_error = None

        try:
            if bbox_source == "generated":
                generated_text, data = generate_bbox(model, processor, image, device, max_new_tokens, prompt)
                raw_box = parse_generated_bbox(generated_text)
                eval_box = normalize_box(raw_box, image.width, image.height, bbox_coord_format)
            else:
                data = make_messages(source, processor)
                eval_box = jitter_box(gt_box, bbox_jitter, rng)
            eval_box_pixels = box_to_pixels(eval_box.tolist(), image.width, image.height)
            eval_bbox_iou = bbox_iou(eval_box_pixels, sample["bbox_2d"])
        except Exception as e:
            if bbox_source != "generated":
                raise
            bbox_parse_success = False
            bbox_parse_error = str(e)
            rows.append(
                {
                    "idx": idx,
                    "image": sample["image"],
                    "mask": sample["mask"],
                    "threshold": threshold,
                    "bbox_source": bbox_source,
                    "bbox_parse_success": False,
                    "bbox_parse_error": bbox_parse_error,
                    "generated_text": generated_text,
                    "prompt_name": prompt_name,
                    "prompt_hash": prompt_hash,
                    "bbox_coord_format": bbox_coord_format,
                    "dice": None,
                    "miou": None,
                    "iou": None,
                    "intersection": None,
                    "union": None,
                    "pred_pixels": None,
                    "gt_pixels": int(gt_orig.sum()),
                    "pred_area_ratio": None,
                    "gt_area_ratio": float(gt_orig.mean()),
                    "bbox_2d": sample["bbox_2d"],
                    "eval_bbox_2d": None,
                    "bbox_iou": None,
                }
            )
            continue

        with torch.no_grad():
            logits = model.predict_masks(
                data["pixel_values"].to(device),
                data["image_grid_thw"].to(device),
                eval_box[None].to(device),
                seg_images=seg_images,
            )
            probs = torch.sigmoid(logits)[0, 0].float().cpu()

        pred_orig = F.interpolate(
            probs[None, None],
            size=gt_orig.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        pred_bin = (pred_orig >= threshold).astype(np.uint8)
        dice, miou, iou, inter, union, pred_pixels, gt_pixels = binary_metrics(pred_bin, gt_orig)
        for item_threshold in threshold_sweep:
            sweep_pred = (pred_orig >= item_threshold).astype(np.uint8)
            sweep_dice, sweep_miou, sweep_iou, sweep_inter, sweep_union, sweep_pred_pixels, _ = binary_metrics(sweep_pred, gt_orig)
            sweep_rows.append(
                {
                    "idx": idx,
                    "image": sample["image"],
                    "threshold": item_threshold,
                    "prompt_name": prompt_name,
                    "prompt_hash": prompt_hash,
                    "bbox_coord_format": bbox_coord_format,
                    "dice": sweep_dice,
                    "miou": sweep_miou,
                    "iou": sweep_iou,
                    "intersection": sweep_inter,
                    "union": sweep_union,
                    "pred_pixels": sweep_pred_pixels,
                    "gt_pixels": gt_pixels,
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
                "bbox_source": bbox_source,
                "bbox_jitter": bbox_jitter,
                "bbox_parse_success": bbox_parse_success,
                "bbox_parse_error": bbox_parse_error,
                "generated_text": generated_text,
                "prompt_name": prompt_name,
                "prompt_hash": prompt_hash,
                "bbox_coord_format": bbox_coord_format,
                "dice": dice,
                "miou": miou,
                "iou": iou,
                "intersection": inter,
                "union": union,
                "pred_pixels": pred_pixels,
                "gt_pixels": gt_pixels,
                "pred_area_ratio": float(pred_bin.mean()),
                "gt_area_ratio": float(gt_orig.mean()),
                "bbox_2d": sample["bbox_2d"],
                "eval_bbox_2d": eval_box_pixels,
                "bbox_iou": eval_bbox_iou,
            }
        )
        vis_items.append(
            {
                "idx": idx,
                "stem": Path(sample["image"]).stem,
                "image": image,
                "gt_mask": gt_orig.astype(bool),
                "pred_mask": pred_bin.astype(bool),
                "box": eval_box_pixels,
                "dice": dice,
            }
        )

    df = pd.DataFrame(rows)
    if max_overlays > 0:
        if overlay_top_k_worst > 0:
            selected_items = sorted(vis_items, key=lambda item: item["dice"])[:overlay_top_k_worst]
        else:
            selected_items = vis_items[:max_overlays]
        for item in selected_items[:max_overlays]:
            overlay = overlay_image(item["image"], item["gt_mask"], item["pred_mask"], item["box"], overlay_alpha)
            overlay.save(overlay_dir / f"{item['idx']:03d}_{item['stem']}.png")
    save_overview(vis_items, output_dir / "overview.png", overlay_alpha, overview_count)
    valid_df = df[df["bbox_parse_success"].fillna(False) & df["iou"].notna()]
    total_inter = int(valid_df["intersection"].sum()) if len(valid_df) else 0
    total_union = int(valid_df["union"].sum()) if len(valid_df) else 0
    summary = {
        "annotation": str(annotation_path),
        "checkpoint": str(checkpoint_path),
        "bbox_source": bbox_source,
        "bbox_jitter": bbox_jitter,
        "bbox_jitter_seed": bbox_jitter_seed,
        "bbox_coord_format": bbox_coord_format,
        "prompt_name": prompt_name,
        "prompt_hash": prompt_hash,
        "prompt_file": prompt_file,
        "prompt": prompt,
        "threshold": threshold,
        "samples": len(rows),
        "evaluated_samples": int(len(valid_df)),
        "bbox_parse_success_rate": float(df["bbox_parse_success"].fillna(False).mean()),
        "dice_mean": float(df["dice"].mean()),
        "dice_median": float(df["dice"].median()),
        "miou_mean": float(df["miou"].mean()),
        "miou_median": float(df["miou"].median()),
        "iou_mean": float(df["iou"].mean()),
        "iou_median": float(df["iou"].median()),
        "ciou": float(total_inter / total_union) if total_union > 0 else None,
        "p_at_0_5": float((valid_df["iou"] >= 0.5).mean()) if len(valid_df) else None,
        "p_at_0_7": float((valid_df["iou"] >= 0.7).mean()) if len(valid_df) else None,
        "p_at_0_9": float((valid_df["iou"] >= 0.9).mean()) if len(valid_df) else None,
        "bbox_iou_mean": float(df["bbox_iou"].mean()),
        "bbox_iou_median": float(df["bbox_iou"].median()),
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
                iou_mean=("iou", "mean"),
                iou_median=("iou", "median"),
                pred_area_ratio_mean=("pred_area_ratio", "mean"),
                gt_area_ratio_mean=("gt_area_ratio", "mean"),
            )
            .sort_values(["dice_mean", "miou_mean"], ascending=False)
        )
        best = threshold_df.iloc[0].to_dict()
        summary["best_threshold"] = float(best["threshold"])
        summary["best_threshold_dice_mean"] = float(best["dice_mean"])
        summary["best_threshold_miou_mean"] = float(best["miou_mean"])
        summary["best_threshold_iou_mean"] = float(best["iou_mean"])
        sweep_df.to_json(output_dir / "threshold_predictions.jsonl", orient="records", lines=True, force_ascii=False)
        threshold_df.to_excel(output_dir / "threshold_sweep.xlsx", index=False)
        (output_dir / "threshold_sweep.json").write_text(
            json.dumps(threshold_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_json(output_dir / "predictions.jsonl", orient="records", lines=True, force_ascii=False)
    df.to_excel(output_dir / "metrics.xlsx", index=False)
    write_failure_cases(rows, output_dir / "failure_cases.json", failure_cases_count)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
