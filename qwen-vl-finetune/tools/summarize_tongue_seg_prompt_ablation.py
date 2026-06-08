import json
from pathlib import Path

import pandas as pd


EVALS = [
    ("val_default_auto", "outputs/tongue_seg_phase2b_eval_val_generated"),
    ("val_default_qwen1000", "outputs/tongue_seg_phase2b_eval_val_generated_qwen1000"),
    ("val_prompt_qwen1000", "outputs/tongue_seg_phase2b_eval_val_generated_prompt_qwen1000"),
    ("val_prompt_pixel_strict", "outputs/tongue_seg_phase2b_eval_val_generated_prompt_pixel"),
    ("test_default_qwen1000", "outputs/tongue_seg_phase2b_eval_test_generated_qwen1000"),
]


def main():
    output_dir = Path("outputs/tongue_seg_p1_prompt_summary")
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
        "bbox_coord_format",
        "prompt_name",
        "prompt_hash",
        "dice_mean",
        "miou_mean",
        "ciou",
        "p_at_0_9",
        "bbox_iou_mean",
        "bbox_parse_success_rate",
        "eval_dir",
    ]
    for col in leading:
        if col not in df.columns:
            df[col] = None
    df = df[leading + [col for col in df.columns if col not in leading]]
    df.to_json(output_dir / "comparison.jsonl", orient="records", lines=True, force_ascii=False)
    df.to_excel(output_dir / "comparison.xlsx", index=False)

    by_run = {row["run"]: row for row in rows}
    conclusion = {
        "best_val_run": "val_default_qwen1000",
        "selected_test_run": "test_default_qwen1000",
        "finding": "Qwen3-VL generated bbox_2d should be interpreted as 0-1000 coordinates for this prompt family.",
        "evidence": {
            "val_default_auto": {
                "bbox_iou_mean": by_run["val_default_auto"]["bbox_iou_mean"],
                "dice_mean": by_run["val_default_auto"]["dice_mean"],
                "p_at_0_9": by_run["val_default_auto"]["p_at_0_9"],
            },
            "val_default_qwen1000": {
                "bbox_iou_mean": by_run["val_default_qwen1000"]["bbox_iou_mean"],
                "dice_mean": by_run["val_default_qwen1000"]["dice_mean"],
                "p_at_0_9": by_run["val_default_qwen1000"]["p_at_0_9"],
            },
            "val_prompt_pixel_strict": {
                "bbox_iou_mean": by_run["val_prompt_pixel_strict"]["bbox_iou_mean"],
                "dice_mean": by_run["val_prompt_pixel_strict"]["dice_mean"],
                "p_at_0_9": by_run["val_prompt_pixel_strict"]["p_at_0_9"],
            },
            "test_default_qwen1000": {
                "bbox_iou_mean": by_run["test_default_qwen1000"]["bbox_iou_mean"],
                "dice_mean": by_run["test_default_qwen1000"]["dice_mean"],
                "miou_mean": by_run["test_default_qwen1000"]["miou_mean"],
                "p_at_0_9": by_run["test_default_qwen1000"]["p_at_0_9"],
            },
        },
        "decision": "Use default prompt with bbox_coord_format=qwen1000. Do not run bbox-only LoRA before new evidence shows a localization problem.",
    }
    (output_dir / "conclusion.json").write_text(
        json.dumps(conclusion, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
