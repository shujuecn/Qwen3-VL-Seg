import json
import subprocess
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback


class TongueSegTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        if isinstance(outputs, dict):
            self._latest_seg_metrics = {}
            for key in ("seg_loss", "bce_loss", "dice_loss", "pred_area_ratio", "gt_area_ratio"):
                value = outputs.get(key)
                if torch.is_tensor(value):
                    self._latest_seg_metrics[key] = float(value.detach().float().cpu())
                elif value is not None:
                    self._latest_seg_metrics[key] = float(value)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        latest = getattr(self, "_latest_seg_metrics", None)
        if latest:
            logs = {**logs, **latest}
        return super().log(logs, *args, **kwargs)


class JsonlLogCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.path = Path(output_dir) / "train_log.jsonl"

    def on_train_begin(self, args, state, control, **kwargs):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if state.global_step == 0:
            self.path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": state.global_step, "epoch": state.epoch}
        row.update(logs)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None
