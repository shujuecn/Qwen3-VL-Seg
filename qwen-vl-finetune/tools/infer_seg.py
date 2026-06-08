import json
import os
import re
import sys
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
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


BASE_MODEL = "/home/zyzd/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct"
CHECKPOINT = "outputs/tongue_seg_phase2b/model.safetensors"
OUTPUT_DIR = "outputs/tongue_seg_infer"
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


def prompt_text(prompt):
    return prompt.replace("<image>\n", "").replace("<image>", "").strip()


def load_prompt(prompt_file, prompt):
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    return prompt


def hash_prompt(prompt):
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]


def build_inputs(processor, image, add_generation_prompt, prompt):
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
        add_generation_prompt=add_generation_prompt,
    )


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def generate_bbox(base_model, processor, image, device, max_new_tokens, prompt):
    inputs = to_device(build_inputs(processor, image, add_generation_prompt=True, prompt=prompt), device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output_ids = base_model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    text = processor.tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)
    return parse_generated_bbox(text), text


def seg_image_tensor(image, mask_size):
    resized = image.resize((mask_size, mask_size), Image.BILINEAR)
    image_np = np.array(resized, dtype=np.float32)
    return torch.from_numpy(image_np).permute(2, 0, 1)[None] / 255.0


def mask_rgba(mask, color, alpha):
    rgba = Image.new("RGBA", mask.shape[::-1], (*color, 0))
    rgba.putalpha(Image.fromarray((mask.astype(np.uint8) * alpha), mode="L"))
    return rgba


def overlay_image(image, pred_mask, box, alpha):
    overlay = Image.alpha_composite(image.convert("RGBA"), mask_rgba(pred_mask, (230, 40, 40), alpha)).convert("RGB")
    ImageDraw.Draw(overlay).rectangle(box, outline=(255, 210, 0), width=3)
    return overlay


def load_checkpoint(path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(path, map_location="cpu")


def main():
    image_arg = arg_value("--image", "")
    if not image_arg:
        raise ValueError("usage: python qwen-vl-finetune/tools/infer_seg.py --image path/to/image.png")

    checkpoint_path = Path(arg_value("--checkpoint", CHECKPOINT))
    config = load_run_config(checkpoint_path)
    base_model_path = arg_value("--model_name_or_path", config.get("model_name_or_path", BASE_MODEL))
    output_dir = Path(arg_value("--output_dir", OUTPUT_DIR))
    mask_size = int(arg_value("--seg_mask_size", config.get("seg_mask_size", 256)))
    threshold = float(arg_value("--threshold", "0.5"))
    overlay_alpha = int(arg_value("--overlay_alpha", "90"))
    max_new_tokens = int(arg_value("--max_new_tokens", "128"))
    box_arg = arg_value("--bbox", "")
    bbox_coord_format = arg_value("--bbox_coord_format", "auto" if box_arg else "qwen1000")
    prompt_file = arg_value("--prompt_file", "")
    prompt = load_prompt(prompt_file, arg_value("--prompt", PROMPT))
    prompt_name = arg_value(
        "--prompt_name",
        Path(prompt_file).stem if prompt_file else ("default" if prompt == PROMPT else "custom"),
    )
    prompt_hash = hash_prompt(prompt)
    device = torch.device(arg_value("--device", "cuda" if torch.cuda.is_available() else "cpu"))

    seg_box_expand = float(arg_value("--seg_box_expand", config.get("seg_box_expand", 0.0)))
    seg_box_alpha = float(arg_value("--seg_box_alpha", config.get("seg_box_alpha", 0.0)))
    seg_use_highres_fusion = bool_arg("--seg_use_highres_fusion", config.get("seg_use_highres_fusion", False))
    seg_refine = bool_arg("--seg_refine", config.get("seg_refine", False))
    seg_use_box_film = bool_arg("--seg_use_box_film", config.get("seg_use_box_film", False))
    seg_box_fourier_bands = int(arg_value("--seg_box_fourier_bands", config.get("seg_box_fourier_bands", 8)))

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_arg)
    image = Image.open(image_path).convert("RGB")
    image.load()

    processor = AutoProcessor.from_pretrained(base_model_path)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        attn_implementation="sdpa",
        dtype=torch.bfloat16 if device.type == "cuda" else None,
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
    model.load_state_dict(load_checkpoint(checkpoint_path), strict=True)
    model.to(device)
    model.eval()

    generated_text = None
    if box_arg:
        box = parse_box(box_arg)
        bbox_source = "argument"
    else:
        box, generated_text = generate_bbox(model.base_model, processor, image, device, max_new_tokens, prompt)
        bbox_source = "generated"

    inputs = to_device(build_inputs(processor, image, add_generation_prompt=True, prompt=prompt), device)
    norm_box = normalize_box(box, image.width, image.height, bbox_coord_format)
    gt_boxes = norm_box[None].to(device)
    seg_images = seg_image_tensor(image, mask_size).to(device)
    with torch.no_grad():
        logits = model.predict_masks(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            gt_boxes,
            seg_images=seg_images,
        )
        probs = torch.sigmoid(logits)[0, 0].float().cpu()

    pred_orig = F.interpolate(
        probs[None, None],
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    pred_mask = pred_orig >= threshold
    box_pixels = box_to_pixels(norm_box.tolist(), image.width, image.height)

    stem = image_path.stem
    mask_path = output_dir / f"{stem}_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    Image.fromarray(pred_mask.astype(np.uint8) * 255).save(mask_path)
    overlay_image(image, pred_mask, box_pixels, overlay_alpha).save(overlay_path)

    result = {
        "image": str(image_path),
        "checkpoint": str(checkpoint_path),
        "bbox_source": bbox_source,
        "bbox_coord_format": bbox_coord_format,
        "bbox_2d": box_pixels,
        "prompt_name": prompt_name,
        "prompt_hash": prompt_hash,
        "prompt_file": prompt_file,
        "prompt": prompt,
        "threshold": threshold,
        "pred_area_ratio": float(pred_mask.mean()),
        "mask": str(mask_path),
        "overlay": str(overlay_path),
    }
    if generated_text is not None:
        result["generated_text"] = generated_text
    (output_dir / f"{stem}_prediction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
