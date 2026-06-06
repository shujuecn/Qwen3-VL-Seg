# Qwen3-VL 舌体分割改造

本项目基于 Qwen3-VL 与 `qwen-vl-finetune`，参考 `materials/2605.07141v1.pdf` 的 Qwen3-VL-Seg 思路，做一个单类舌体二值分割的轻量可训练版本。

当前实现是 Phase 1：先跑通数据、训练入口、最小 mask head 和 smoke test。它不是完整复现论文的 17M box-guided decoder，也不做开放世界 referring segmentation。

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
  --output_dir outputs/tongue_seg_phase1
```

## 可行性验证

不加载大模型的本地验证：

```bash
python -m py_compile \
  qwen-vl-finetune/tools/build_tongue_seg_json.py \
  qwen-vl-finetune/qwenvl/data/__init__.py \
  qwen-vl-finetune/qwenvl/data/data_processor.py \
  qwen-vl-finetune/qwenvl/train/argument.py \
  qwen-vl-finetune/qwenvl/train/train_qwen.py \
  qwen-vl-finetune/qwenvl/model/__init__.py \
  qwen-vl-finetune/qwenvl/model/qwen3vl_seg.py
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

4B 10 step 和长训练可继续用上面的本地路径执行。

## 当前实现位置

- 数据生成：`qwen-vl-finetune/tools/build_tongue_seg_json.py`
- 数据注册：`qwen-vl-finetune/qwenvl/data/__init__.py`
- mask/bbox 读取与 collator：`qwen-vl-finetune/qwenvl/data/data_processor.py`
- 分割 wrapper：`qwen-vl-finetune/qwenvl/model/qwen3vl_seg.py`
- 训练入口：`qwen-vl-finetune/qwenvl/train/train_qwen.py`

## 下一步

Phase 2 再考虑加入论文里的关键增强：

- bbox soft gate 扩大 15% 与 sigmoid soft boundary
- shallow image stem
- high-resolution pixel fusion
- mask-aware refinement
- Dice/mIoU 评估脚本与 overlay 输出
