# Qwen3-VL 舌体分割改造

本项目基于 Qwen3-VL 与 `qwen-vl-finetune`，参考 `materials/2605.07141v1.pdf` 的 Qwen3-VL-Seg 思路，做一个单类舌体二值分割的轻量可训练版本。

当前实现是 Phase 1：先跑通数据、训练入口、最小 mask head 和 smoke test。它不是完整复现论文的 17M box-guided decoder，也不做开放世界 referring segmentation。

Phase 2A 已加入 soft bbox gate 和高分辨率 RGB 浅层融合。为了保留 Phase 1 baseline，Phase 2 功能默认不开启，需要在命令中显式传参。

原始上游 README 已保留为 `README_QWEN3VL_ORIGINAL.md`。

## 思路

Qwen3-VL-Seg 的核心启发是：不要只把 VLM 特征直接上采样成 mask，而是把 bbox 当作结构先验，引导分割 decoder。

Phase 1 采用最小可运行版本：

- 数据为单图单 mask：`image + binary mask`。
- bbox 从 GT mask 自动计算。
- Qwen3-VL 主体冻结。
- 新增轻量 `TongueMaskHead` 训练舌体 mask。
- 使用 Qwen3-VL `get_image_features(...)` 的视觉 token 作为分割特征。
- 将 bbox rasterize 成 gate，与视觉特征一起输入 mask head。
- loss 为 `seg_loss_weight * (BCE + Dice)`；当前 base 在 `no_grad` 下运行，不计算 LM loss，实际可训练梯度只进入 mask head。

详细改造计划见 `doc/改造计划.md`。

## Git 数据策略

图像数据不进入 Git：

- `data/TongeImageDataset/origin_Image/`
- `data/TongeImageDataset/origin_GT/`

JSON 划分文件可以进入 Git：

- `data/TongeImageDataset/train.json`
- `data/TongeImageDataset/val.json`
- `data/TongeImageDataset/test.json`

这样仓库记录数据划分和训练接口，但不提交原始图片/mask。

## 数据准备

数据目录应为：

```text
data/TongeImageDataset/
  origin_Image/
    1.png
    ...
  origin_GT/
    1.png
    ...
```

注意：当前 `origin_Image/*.png` 与 `origin_GT/*.png` 的扩展名是 `.png`，但文件内容实际是 BMP。数据管线已用 PIL 读取并转为 RGB，避免 transformers/torchvision 按 PNG 解码时报错。

生成训练 JSON：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/tools/build_tongue_seg_json.py
```

当前默认固定划分：

- train: 240
- val: 30
- test: 30

每条 JSON 包含：

- `image`
- `mask`
- `bbox_2d`
- `label`
- Qwen SFT 格式的 `conversations`

## 训练

Phase 1 只支持 dense Qwen3-VL，不支持 Qwen3-VL MoE、LoRA、`data_flatten` 或 `data_packing`。

当前环境的 `transformers + kernels` 组合会在导入 hub kernels 时触发 `Either a revision or a version must be specified`。训练脚本已在进程内屏蔽 broken `kernels` 包，并在没有 `flash_attn` 时自动使用 `sdpa`。

VS Code debug 配置已写入 `.vscode/launch.json`，包含：

- `Tongue Seg Train 2B Smoke`
- `Tongue Seg Train 4B Smoke`
- `Tongue Seg Train 4B Phase1`
- `Tongue Seg Eval Val`
- `Tongue Seg Train 2B Phase2 Smoke`
- `Tongue Seg Train 4B Phase2 Smoke`
- `Tongue Seg Train 4B Phase2`
- `Tongue Seg Eval Val Phase2`
- `Tongue Seg Eval Test Phase2`

使用 VS Code 运行这些配置时，建议选择 `torch` conda 环境对应的 Python 解释器。

本机已用 2B 模型跑通 1 step smoke test。优先用 2B 验证链路：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/qwenvl/train/train_qwen.py \
  --model_name_or_path /home/zyzd/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct \
  --dataset_use tongue_seg \
  --seg_enable True \
  --seg_mask_size 256 \
  --seg_loss_weight 1.0 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps 1 \
  --learning_rate 1e-4 \
  --output_dir outputs/tongue_seg_2b_phase1_smoke \
  --save_strategy no \
  --logging_steps 1
```

本机也已用本地 4B 模型跑通 1 step smoke test。建议优先使用本地 ModelScope 路径，避免 HuggingFace ID 触发额外下载：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/qwenvl/train/train_qwen.py \
  --model_name_or_path /home/zyzd/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct \
  --dataset_use tongue_seg \
  --seg_enable True \
  --seg_mask_size 256 \
  --seg_loss_weight 1.0 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps 1 \
  --learning_rate 1e-4 \
  --output_dir outputs/tongue_seg_4b_phase1_smoke \
  --save_strategy no \
  --logging_steps 1
```

4B 的 10 step smoke test：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/qwenvl/train/train_qwen.py \
  --model_name_or_path /home/zyzd/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct \
  --dataset_use tongue_seg \
  --seg_enable True \
  --seg_mask_size 256 \
  --seg_loss_weight 1.0 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps 10 \
  --learning_rate 1e-4 \
  --output_dir outputs/tongue_seg_phase1_smoke
```

较长训练可从下面开始：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/qwenvl/train/train_qwen.py \
  --model_name_or_path /home/zyzd/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct \
  --dataset_use tongue_seg \
  --seg_enable True \
  --seg_mask_size 256 \
  --seg_loss_weight 1.0 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 30 \
  --learning_rate 1e-4 \
  --output_dir outputs/tongue_seg_phase1 \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 3
```

Phase 2A 在 Phase 1 基础上开启：

- `seg_box_expand=0.15`：bbox 宽高外扩 15%。
- `seg_box_alpha=20.0`：sigmoid soft boundary 系数。
- `seg_use_highres_fusion=True`：启用 `256x256` RGB shallow image stem。

2B Phase 2 smoke：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/qwenvl/train/train_qwen.py \
  --model_name_or_path /home/zyzd/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct \
  --dataset_use tongue_seg \
  --seg_enable True \
  --seg_mask_size 256 \
  --seg_loss_weight 1.0 \
  --seg_box_expand 0.15 \
  --seg_box_alpha 20.0 \
  --seg_use_highres_fusion True \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps 3 \
  --learning_rate 1e-4 \
  --output_dir outputs/tongue_seg_2b_phase2_smoke \
  --save_strategy no \
  --logging_steps 1
```

4B Phase 2 正式训练：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/qwenvl/train/train_qwen.py \
  --model_name_or_path /home/zyzd/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct \
  --dataset_use tongue_seg \
  --seg_enable True \
  --seg_mask_size 256 \
  --seg_loss_weight 1.0 \
  --seg_box_expand 0.15 \
  --seg_box_alpha 20.0 \
  --seg_use_highres_fusion True \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 30 \
  --learning_rate 1e-4 \
  --output_dir outputs/tongue_seg_phase2 \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 3
```

`seg_enable=True` 时，如果未显式传入日志参数，训练脚本会默认使用：

- `logging_steps=10`
- `save_steps=100`
- `save_total_limit=3`

训练会额外输出：

- `run_config.json`：运行配置、环境版本、git commit
- `train_log.jsonl`：逐次日志记录，包含 `seg_loss`、`bce_loss`、`dice_loss`、预测/GT 面积比例

输出清理策略：

- 保留正式训练目录的 `model.safetensors`、`run_config.json`、`train_log.jsonl`、tokenizer/processor 配置。
- 保留正式评估目录的 `summary.json`、`metrics.xlsx`、`predictions.jsonl` 和必要 overlay。
- 训练中间 `checkpoint-*`、smoke 输出、compat 评估输出可以删除。

## 可行性验证

不加载大模型的本地验证：

```bash
python -m py_compile \
  qwen-vl-finetune/tools/build_tongue_seg_json.py \
  qwen-vl-finetune/qwenvl/data/__init__.py \
  qwen-vl-finetune/qwenvl/data/data_processor.py \
  qwen-vl-finetune/qwenvl/train/argument.py \
  qwen-vl-finetune/qwenvl/train/train_qwen.py \
  qwen-vl-finetune/qwenvl/train/seg_trainer.py \
  qwen-vl-finetune/qwenvl/model/__init__.py \
  qwen-vl-finetune/qwenvl/model/qwen3vl_seg.py \
  qwen-vl-finetune/tools/eval_tongue_seg.py
```

已验证：

- JSON 生成脚本可输出 240/30/30 划分。
- `TongueMaskHead` 输出 shape 正确。
- fake Qwen3-VL wrapper forward/backward 可运行。
- mask head 有梯度，base model 无梯度。
- 本地 Qwen3-VL-2B-Instruct 真实 1 step smoke test 通过，loss 正常输出。
- 本地 Qwen3-VL-4B-Instruct 真实 1 step smoke test 通过，loss 正常输出。
- 本地 Qwen3-VL-4B-Instruct 真实 10 step smoke test 通过，checkpoint 保存通过。
- Phase 1 checkpoint 的 `model.safetensors` 只包含 `mask_head.*` 权重，不保存 frozen Qwen base。
- 2B debug smoke 已验证 `train_log.jsonl` 每 step 记录分项 loss。
- `outputs/tongue_seg_phase1/model.safetensors` 在 val split 上评估：Dice mean `0.9741`，mIoU mean `0.9497`。
- Phase 2A 2B/4B smoke test 通过，checkpoint 只包含 `mask_head.*`。
- `outputs/tongue_seg_phase2/model.safetensors` 在 val split 上评估：Dice mean `0.9741`，mIoU mean `0.9498`。
- `outputs/tongue_seg_phase2/model.safetensors` 在 test split 上评估：Dice mean `0.9726`，mIoU mean `0.9469`。
- `outputs/tongue_seg_phase2b/model.safetensors` 在 val split 上评估：Dice mean `0.9764`，mIoU mean `0.9541`。
- `outputs/tongue_seg_phase2b/model.safetensors` 在 test split 上评估：Dice mean `0.9753`，mIoU mean `0.9520`。

4B 10 step 和长训练可继续用上面的本地路径执行。

## 评估

当前最佳版本是 Phase 2B。评估默认使用 JSON 中的 GT bbox，只验证 box-guided mask head 能力：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/tools/eval_tongue_seg.py \
  --model_name_or_path /home/zyzd/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct \
  --checkpoint outputs/tongue_seg_phase2b/model.safetensors \
  --annotation data/TongeImageDataset/val.json \
  --output_dir outputs/tongue_seg_phase2b_eval_val \
  --seg_mask_size 256 \
  --seg_box_expand 0.15 \
  --seg_box_alpha 20.0 \
  --seg_use_highres_fusion True \
  --seg_refine True \
  --max_overlays 5 \
  --overlay_top_k_worst 5 \
  --overlay_alpha 85 \
  --overview_count 5
```

输出：

- `summary.json`
- `metrics.xlsx`
- `predictions.jsonl`
- `overlays/*.png`：最差样本透明 overlay
- `overview.png`：多子图总览，适合放进演示文档

test split 评估：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/tools/eval_tongue_seg.py \
  --model_name_or_path /home/zyzd/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct \
  --checkpoint outputs/tongue_seg_phase2b/model.safetensors \
  --annotation data/TongeImageDataset/test.json \
  --output_dir outputs/tongue_seg_phase2b_eval_test \
  --seg_mask_size 256 \
  --seg_box_expand 0.15 \
  --seg_box_alpha 20.0 \
  --seg_use_highres_fusion True \
  --seg_refine True \
  --max_overlays 5 \
  --overlay_top_k_worst 5 \
  --overlay_alpha 85 \
  --overview_count 5
```

可视化示例：

![Phase 2B val overview](outputs/tongue_seg_phase2b_eval_val/overview.png)

![Phase 2B test overview](outputs/tongue_seg_phase2b_eval_test/overview.png)

单张透明 overlay 示例：

![Worst test overlay](outputs/tongue_seg_phase2b_eval_test/overlays/006_277.png)

Phase 2A threshold sweep 已验证：

- val 最优阈值仍为 `0.50`。
- test 最优阈值为 `0.45`，但 Dice mean 只从 `0.972597` 到 `0.972675`，提升很小。
- 结论：当前主要瓶颈不是二值化阈值，而是边界 refinement。

Phase 2B 增加 mask-aware refinement，默认不开启。4B 正式训练命令：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/qwenvl/train/train_qwen.py \
  --model_name_or_path /home/zyzd/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct \
  --dataset_use tongue_seg \
  --seg_enable True \
  --seg_mask_size 256 \
  --seg_loss_weight 1.0 \
  --seg_box_expand 0.15 \
  --seg_box_alpha 20.0 \
  --seg_use_highres_fusion True \
  --seg_refine True \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 30 \
  --learning_rate 1e-4 \
  --output_dir outputs/tongue_seg_phase2b \
  --logging_steps 10 \
  --save_steps 300 \
  --save_total_limit 1
```

当前对比结果：

| phase | split | Dice mean | mIoU mean |
| --- | --- | ---: | ---: |
| Phase 1 | val | 0.9741 | 0.9497 |
| Phase 1 | test | 0.9713 | 0.9444 |
| Phase 2A | val | 0.9741 | 0.9498 |
| Phase 2A | test | 0.9726 | 0.9469 |
| Phase 2B | val | 0.9764 | 0.9541 |
| Phase 2B | test | 0.9753 | 0.9520 |

Phase 2B 是当前最好版本。它相比 Phase 2A 在 val/test 都有明确提升，但 test 最差样本仍是 `origin_Image/277.png`，Dice `0.9265`，主要表现为大舌体边缘欠分割。

## 当前实现位置

- 数据生成：`qwen-vl-finetune/tools/build_tongue_seg_json.py`
- 数据注册：`qwen-vl-finetune/qwenvl/data/__init__.py`
- mask/bbox 读取与 collator：`qwen-vl-finetune/qwenvl/data/data_processor.py`
- 分割 wrapper：`qwen-vl-finetune/qwenvl/model/qwen3vl_seg.py`
- 训练入口：`qwen-vl-finetune/qwenvl/train/train_qwen.py`
- 评估脚本：`qwen-vl-finetune/tools/eval_tongue_seg.py`

## 下一步

Phase 2A 已实现：

- bbox soft gate 扩大 15% 与 sigmoid soft boundary
- shallow image stem
- high-resolution pixel fusion

Phase 2B 已实现可选 mask-aware refinement，并已完成正式训练和 val/test 评估。下一步优先针对 `origin_Image/277.png` 这类大舌体边界欠分割样本做边界诊断，再考虑加入边界 loss 或 bbox 边缘采样。
