import json
from pathlib import Path

import pandas as pd


EVALS = [
    ("val_gt", "outputs/tongue_seg_phase2b_eval_val"),
    ("test_gt", "outputs/tongue_seg_phase2b_eval_test"),
    ("val_jitter_005", "outputs/tongue_seg_phase2b_eval_val_jitter_005"),
    ("test_jitter_005", "outputs/tongue_seg_phase2b_eval_test_jitter_005"),
    ("val_jitter_010", "outputs/tongue_seg_phase2b_eval_val_jitter_010"),
    ("test_jitter_010", "outputs/tongue_seg_phase2b_eval_test_jitter_010"),
    ("val_jitter_015", "outputs/tongue_seg_phase2b_eval_val_jitter_015"),
    ("test_jitter_015", "outputs/tongue_seg_phase2b_eval_test_jitter_015"),
    ("val_generated_auto", "outputs/tongue_seg_phase2b_eval_val_generated"),
    ("test_generated_auto", "outputs/tongue_seg_phase2b_eval_test_generated"),
    ("val_generated_qwen1000", "outputs/tongue_seg_phase2b_eval_val_generated_qwen1000"),
    ("test_generated_qwen1000", "outputs/tongue_seg_phase2b_eval_test_generated_qwen1000"),
]


def main():
    output_dir = Path("outputs/tongue_seg_next_baseline_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, directory in EVALS:
        summary_path = Path(directory) / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        row["run"] = name
        row["eval_dir"] = directory
        rows.append(row)

    df = pd.DataFrame(rows)
    leading = [
        "run",
        "annotation",
        "bbox_source",
        "bbox_jitter",
        "bbox_coord_format",
        "prompt_name",
        "prompt_hash",
        "dice_mean",
        "miou_mean",
        "ciou",
        "p_at_0_5",
        "p_at_0_7",
        "p_at_0_9",
        "bbox_iou_mean",
        "bbox_parse_success_rate",
        "pred_area_ratio_mean",
        "gt_area_ratio_mean",
        "eval_dir",
    ]
    for col in leading:
        if col not in df.columns:
            df[col] = None
    df = df[leading + [col for col in df.columns if col not in leading]]
    df.to_json(output_dir / "comparison.jsonl", orient="records", lines=True, force_ascii=False)
    df.to_excel(output_dir / "comparison.xlsx", index=False)

    by_run = {row["run"]: row for row in rows}
    test_gt = by_run["test_gt"]
    test_generated_auto = by_run["test_generated_auto"]
    test_generated_qwen1000 = by_run["test_generated_qwen1000"]
    test_jitter_010 = by_run["test_jitter_010"]
    test_jitter_015 = by_run["test_jitter_015"]

    conclusion = {
        "baseline_model": "outputs/tongue_seg_phase2b/model.safetensors",
        "baseline_rebuilt": True,
        "phase2b_gt_reproduced": (
            test_gt["dice_mean"] >= 0.9730
            and test_gt["p_at_0_9"] >= 0.9666
            and test_gt["bbox_iou_mean"] == 1.0
        ),
        "main_bottleneck": "bbox_coordinate_format_was_misinterpreted",
        "generated_qwen1000_meets_next_stage_targets": (
            test_generated_qwen1000["bbox_iou_mean"] >= 0.50
            and test_generated_qwen1000["dice_mean"] >= 0.93
            and test_generated_qwen1000["miou_mean"] >= 0.87
            and test_generated_qwen1000["p_at_0_9"] >= 0.35
        ),
        "evidence": {
            "test_gt": {
                "dice_mean": test_gt["dice_mean"],
                "miou_mean": test_gt["miou_mean"],
                "ciou": test_gt["ciou"],
                "p_at_0_9": test_gt["p_at_0_9"],
            },
            "test_jitter_010": {
                "bbox_iou_mean": test_jitter_010["bbox_iou_mean"],
                "dice_mean": test_jitter_010["dice_mean"],
                "p_at_0_9": test_jitter_010["p_at_0_9"],
            },
            "test_jitter_015": {
                "bbox_iou_mean": test_jitter_015["bbox_iou_mean"],
                "dice_mean": test_jitter_015["dice_mean"],
                "p_at_0_9": test_jitter_015["p_at_0_9"],
            },
            "test_generated_auto_old_interpretation": {
                "bbox_parse_success_rate": test_generated_auto["bbox_parse_success_rate"],
                "bbox_iou_mean": test_generated_auto["bbox_iou_mean"],
                "dice_mean": test_generated_auto["dice_mean"],
                "miou_mean": test_generated_auto["miou_mean"],
                "p_at_0_9": test_generated_auto["p_at_0_9"],
            },
            "test_generated_qwen1000": {
                "bbox_parse_success_rate": test_generated_qwen1000["bbox_parse_success_rate"],
                "bbox_iou_mean": test_generated_qwen1000["bbox_iou_mean"],
                "dice_mean": test_generated_qwen1000["dice_mean"],
                "miou_mean": test_generated_qwen1000["miou_mean"],
                "p_at_0_9": test_generated_qwen1000["p_at_0_9"],
            },
        },
        "next_step": "Use qwen1000 as generated-bbox coordinate format; bbox-only LoRA is not justified before new evidence.",
    }
    (output_dir / "conclusion.json").write_text(
        json.dumps(conclusion, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
