# Qwen3-VL 舌体分割复现

本项目基于 Qwen3-VL 与 `qwen-vl-finetune`，参考 `materials/2605.07141v1.pdf` 的 Qwen3-VL-Seg 思路，在 300 张舌象单类二值分割数据上做轻量复现。

当前结论：

- **主模型保留 Phase 2B**：GT bbox 上限最高，作为当前可复现结果。
- **R1-R3 已补齐论文式评估**：`cIoU`、`P@0.5/0.7/0.9`、bbox jitter、generated-bbox 端到端评估。
- **P1 已修正 generated-bbox 坐标制**：Qwen3-VL 默认生成的 `bbox_2d` 更符合 0-1000 坐标；旧评估按像素坐标解析会系统性低估端到端表现。
- **R4 BoxFiLM 已实现但不推荐替代 Phase 2B**：GT/jitter 主指标低于 Phase 2B，按验收规则属于负向消融。
- **暂不做 bbox LoRA / R5 / R6**：修正 qwen1000 坐标后 generated-bbox 已接近 GT-bbox 上限，继续堆 decoder 或 LoRA 暂无证据支撑。

详细路线与验收依据见：

- `doc/Qwen3-VL-Seg论文复现指导计划.md`
- `outputs/tongue_seg_next_baseline_summary/conclusion.json`
- `outputs/tongue_seg_p1_prompt_summary/conclusion.json`
- `doc/qwen1000修正后模型表现优化任务计划.md`

## 数据

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

当前固定划分：

- train: 240
- val: 30
- test: 30

重新生成 JSON：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/tools/build_tongue_seg_json.py
```

图像和 mask 不进 Git；`train.json`、`val.json`、`test.json` 可进 Git，用于固定划分。

## 当前模型结构

Phase 2B 使用：

- 冻结 Qwen3-VL 4B base。
- 只训练 `mask_head.*`。
- bbox 从 GT mask 或 generated bbox 得到，并作为 box gate。
- `seg_box_expand=0.15`
- `seg_box_alpha=20.0`
- `seg_use_highres_fusion=True`
- `seg_refine=True`
- loss = `BCE + Dice`

训练保存策略：

- 只保存 `mask_head.*` 到 `model.safetensors`，不保存 4B base。
- `run_config.json` 记录模型路径、seg 参数、git commit 和环境版本。
- `train_log.jsonl` 记录 `seg_loss`、`bce_loss`、`dice_loss`、预测/GT 面积比例。

R4 BoxFiLM 额外加入：

- normalized bbox -> Fourier PE -> MLP
- 以 FiLM 形式调制 mask feature
- 参数量小幅增加，checkpoint 从约 5.17 MB 到约 5.46 MB

R4 只作为消融记录，不作为当前推荐模型。

## 训练

推荐使用本地 ModelScope 路径，避免触发额外下载：

```text
/home/zyzd/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct
```

正式训练 Phase 2B：

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
  --learning_rate 1e-4 \
  --num_train_epochs 30 \
  --output_dir outputs/tongue_seg_phase2b \
  --logging_steps 10 \
  --save_steps 300 \
  --save_total_limit 1
```

快速 smoke test 可以用更短步数：

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
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_steps 10 \
  --output_dir outputs/tongue_seg_phase2b_smoke \
  --save_strategy no \
  --logging_steps 1
```

R4 BoxFiLM 消融训练命令如下，仅在需要复现实验对比时运行：

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
  --seg_use_box_film True \
  --seg_box_fourier_bands 8 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 30 \
  --output_dir outputs/tongue_seg_phase2b_boxfilm \
  --logging_steps 10 \
  --save_steps 300 \
  --save_total_limit 1
```

## VS Code 调试学习入口

`.vscode/launch.json` 已按当前方案整理成 00-08 的学习顺序：

- `00 Build Tongue Seg JSON`：生成/检查 240/30/30 数据划分和 `bbox_2d`。
- `01 Learn Phase2B Forward 4B One Step`：只跑 1 step，适合断点看 Qwen-VL 特征如何进入 mask head。
- `02 Train Phase2B 4B Full`：正式训练当前推荐模型。
- `03 Eval Phase2B Val GT BBox`：验证集 GT bbox 上限评估。
- `04 Eval Phase2B Test GT BBox`：测试集 GT bbox 上限评估。
- `05 Eval Phase2B Test Jitter 0.10`：观察 bbox 不准时 mask head 鲁棒性。
- `06 Eval Phase2B Test Generated BBox`：真实端到端路径，先让 Qwen-VL 生成 bbox，再预测 mask。
- `07 Infer Single Image Generated BBox`：单张图片推理示例。
- `08 Ablation R4 BoxFiLM One Step`：只用于学习 bbox Fourier/FiLM 消融，不是推荐模型。

推荐断点位置：

- `qwen-vl-finetune/qwenvl/data/data_processor.py`：看原图、mask、bbox 如何变成训练 batch。
- `qwen-vl-finetune/qwenvl/model/qwen3vl_seg.py` 的 `Qwen3VLSegForConditionalGeneration.forward(...)`：看 Qwen-VL frozen base 如何提供视觉 token。
- `qwen-vl-finetune/qwenvl/model/qwen3vl_seg.py` 的 `TongueMaskHead.forward(...)`：看 bbox gate、高分辨率 RGB 分支和 refinement。
- `qwen-vl-finetune/tools/eval_tongue_seg.py` 的 generated-bbox 分支：看 Qwen-VL 如何先生成 `bbox_2d`，再交给 mask head。

## 评估

`eval_tongue_seg.py` 会从 checkpoint 同级 `run_config.json` 自动读取 seg 结构参数。评估 Phase 2B val：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/tools/eval_tongue_seg.py \
  --checkpoint outputs/tongue_seg_phase2b/model.safetensors \
  --annotation data/TongeImageDataset/val.json \
  --output_dir outputs/tongue_seg_phase2b_eval_val \
  --max_overlays 20 \
  --overlay_top_k_worst 20 \
  --overview_count 20
```

评估 test：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/tools/eval_tongue_seg.py \
  --checkpoint outputs/tongue_seg_phase2b/model.safetensors \
  --annotation data/TongeImageDataset/test.json \
  --output_dir outputs/tongue_seg_phase2b_eval_test \
  --max_overlays 20 \
  --overlay_top_k_worst 20 \
  --overview_count 20
```

评估 bbox jitter，例如 test 0.10：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/tools/eval_tongue_seg.py \
  --checkpoint outputs/tongue_seg_phase2b/model.safetensors \
  --annotation data/TongeImageDataset/test.json \
  --output_dir outputs/tongue_seg_phase2b_eval_test_jitter_010 \
  --bbox_jitter 0.10 \
  --bbox_jitter_seed 0 \
  --max_overlays 20 \
  --overlay_top_k_worst 20 \
  --overview_count 20
```

评估 generated-bbox 端到端路径：

```bash
conda run --no-capture-output -n torch python qwen-vl-finetune/tools/eval_tongue_seg.py \
  --checkpoint outputs/tongue_seg_phase2b/model.safetensors \
  --annotation data/TongeImageDataset/test.json \
  --output_dir outputs/tongue_seg_phase2b_eval_test_generated_qwen1000 \
  --bbox_source generated \
  --bbox_coord_format qwen1000 \
  --max_overlays 20 \
  --overlay_top_k_worst 20 \
  --overview_count 20
```

评估输出：

- `summary.json`：均值指标、cIoU、P@0.5/0.7/0.9、bbox IoU
- `metrics.xlsx`：每样本指标
- `predictions.jsonl`：逐样本记录，generated 模式包含原始生成文本
- `failure_cases.json`：最差样本摘要
- `overlays/*.png` 和 `overview.png`：可视化检查

## 当前结果

Phase 2B GT-bbox 上限：

| split | Dice mean | mIoU mean | cIoU | P@0.9 |
| --- | ---: | ---: | ---: | ---: |
| val | 0.9766 | 0.9545 | 0.9549 | 1.0000 |
| test | 0.9749 | 0.9512 | 0.9529 | 0.9667 |

generated-bbox 端到端，按 Qwen 0-1000 坐标解析：

| split | bbox IoU mean | Dice mean | mIoU mean | P@0.9 |
| --- | ---: | ---: | ---: | ---: |
| val | 0.9442 | 0.9756 | 0.9526 | 0.9667 |
| test | 0.9397 | 0.9733 | 0.9485 | 0.9333 |

旧的 generated-bbox 评估把 0-1000 坐标按像素坐标解释，test bbox IoU 只有 `0.3621`，Dice `0.9201`。修正为 `--bbox_coord_format qwen1000` 后，test bbox IoU 升到 `0.9397`，Dice 升到 `0.9733`。结论：主要问题是坐标制解析，不是 Qwen3-VL 定位能力不足。

R4 BoxFiLM 对比结论：

- GT-bbox val Dice 从 `0.9764` 降到 `0.9753`，P@0.9 从 `1.0` 降到 `0.9667`。
- GT-bbox test Dice 从 `0.9753` 降到 `0.9750`，P@0.9 持平 `0.9667`。
- 旧坐标解释下 generated-bbox test Dice 从 `0.9159` 升到 `0.9221`，但 bbox IoU 不变。
- 正确 qwen1000 坐标下，Phase 2B 已满足下一阶段端到端目标；R4 不替代 Phase 2B。

## 输出保留策略

建议保留：

- `outputs/tongue_seg_phase2b/`
- `outputs/tongue_seg_phase2b_eval_val/`
- `outputs/tongue_seg_phase2b_eval_test/`
- `outputs/tongue_seg_phase2b_eval_val_generated_qwen1000/`
- `outputs/tongue_seg_phase2b_eval_test_generated_qwen1000/`
- `outputs/tongue_seg_next_baseline_summary/`
- `outputs/tongue_seg_p1_prompt_summary/`

可删除：

- smoke 输出，如 `outputs/*smoke*`
- 训练中间 `checkpoint-*`
- 已汇总后的旧 R4 详细 eval 目录：`outputs/tongue_seg_phase2b_boxfilm_eval_*`
- 过期临时对比文件

不要删除：

- `outputs/tongue_seg_phase2b/model.safetensors`
- `outputs/tongue_seg_phase2b/run_config.json`
- `outputs/tongue_seg_phase2b/train_log.jsonl`
- summary 目录里的 `comparison.xlsx`、`comparison.jsonl`、`conclusion.json`

## 本地验证

不加载大模型的语法检查：

```bash
conda run -n torch python -m py_compile \
  qwen-vl-finetune/qwenvl/model/qwen3vl_seg.py \
  qwen-vl-finetune/qwenvl/train/argument.py \
  qwen-vl-finetune/qwenvl/train/train_qwen.py \
  qwen-vl-finetune/tools/eval_tongue_seg.py \
  qwen-vl-finetune/tools/infer_seg.py
```

当前 P0/P1 审计已写入：

```text
outputs/tongue_seg_next_baseline_summary/conclusion.json
outputs/tongue_seg_p1_prompt_summary/conclusion.json
```
