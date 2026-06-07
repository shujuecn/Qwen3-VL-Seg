# Qwen3-VL-Seg 论文复现指导计划

## 目标

这份文档是在不覆盖 `doc/改造计划.md` 的前提下，对后续复现工作做一份更贴近论文逻辑的工程指导。它参考：

- 当前仓库 `README.md`
- 现有 `doc/改造计划.md`
- `materials/2605.07141v1.pdf`
- 当前代码中的舌体分割实现

当前目标不是完整复现论文的开放世界 referring segmentation，而是在本仓库已有 300 张舌象数据、单卡 RTX 4090 约 24GiB 显存、31GiB 内存的约束下，把论文中真正可落地、对舌体二值分割有帮助的代码逻辑梳理清楚，形成后续增量复现路线。

## 当前仓库状态

仓库已经从 Qwen3-VL / `qwen-vl-finetune` 改造成一个轻量舌体二值分割工程。

已有数据：

- `data/TongeImageDataset/origin_Image/`：300 张舌象图像
- `data/TongeImageDataset/origin_GT/`：300 张二值 mask
- `train.json` / `val.json` / `test.json`：固定 240 / 30 / 30 划分
- 每条样本包含 `image`、`mask`、`bbox_2d`、`label`、`conversations`
- bbox 由 GT mask 自动计算，训练和当前评估默认使用 GT bbox

已有核心代码：

- 数据生成：`qwen-vl-finetune/tools/build_tongue_seg_json.py`
- 数据注册：`qwen-vl-finetune/qwenvl/data/__init__.py`
- mask/bbox 读取：`qwen-vl-finetune/qwenvl/data/data_processor.py`
- 分割 wrapper：`qwen-vl-finetune/qwenvl/model/qwen3vl_seg.py`
- 分割 trainer：`qwen-vl-finetune/qwenvl/train/seg_trainer.py`
- 训练入口：`qwen-vl-finetune/qwenvl/train/train_qwen.py`
- 评估脚本：`qwen-vl-finetune/tools/eval_tongue_seg.py`

已有结果：

| 版本 | split | Dice mean | mIoU mean |
| --- | --- | ---: | ---: |
| Phase 1 | val | 0.9741 | 0.9497 |
| Phase 1 | test | 0.9713 | 0.9444 |
| Phase 2A | val | 0.9741 | 0.9498 |
| Phase 2A | test | 0.9726 | 0.9469 |
| Phase 2B | val | 0.9764 | 0.9541 |
| Phase 2B | test | 0.9753 | 0.9520 |

当前最佳是 Phase 2B：

- `seg_box_expand=0.15`
- `seg_box_alpha=20.0`
- `seg_use_highres_fusion=True`
- `seg_refine=True`

需要注意：这些指标主要验证的是 `GT bbox + box-guided mask head` 的上限，不等价于论文里的完整端到端开放世界分割能力。

## 论文原始逻辑

论文核心问题是：MLLM 可以输出开放世界 bbox，但 bbox 太粗，不能满足像素级边界要求。Qwen3-VL-Seg 的做法是把 MLLM 预测的 bbox 当作结构先验，贯穿 mask decoder，而不是把 bbox 当作最终输出。

论文 forward 路径可以抽象为：

```text
image + referring instruction
  -> Qwen3-VL
    -> multi-scale visual features {F_vis^l}
    -> multimodal visual embeddings T_mm
    -> segmentation token feature T_seg
    -> grounded bbox B_box
  -> box-guided mask decoder
    -> mask logits
    -> final binary mask
```

decoder 输入包含五类信息：

- `F_vis^l`：视觉编码器中间层和顶层特征，用于补空间细节
- `T_mm`：经过多模态对齐的视觉 token，用于保留语言条件下的语义
- `T_seg`：文本输出中 `<mask_token>` 对应的隐藏状态，用于绑定当前实例/表达式
- `B_box`：MLLM 输出的 bbox，用作空间先验
- `I`：原始图像，用于浅层高分辨率纹理补充

论文 decoder 有四个模块。

### 1. Multi-scale Spatial Feature Injection

论文认为高层 MLLM 特征太粗，不适合边界分割，所以从视觉 encoder 中取多层特征。每层先用 `1x1 Conv` 投影到 decoder hidden dim，再走一个轻量 depthwise branch：

```text
X0 = Conv1x1(F_vis^l)
F_tilde = X0 + s * GELU(GroupNorm(DWConv(X0)))
```

其中 `s` 是可学习标量，初始化为 `1e-3`。这个细节很重要：它让新加的空间 adapter 初始接近 identity，避免一开始破坏预训练视觉特征。

多层 `F_tilde` 会 concat 后再卷积融合，形成 `F_fuse`。同时 `T_mm` 被投影并 reshape 成二维特征图，最终 memory 是：

```text
F_mem = reshape(W_mm(T_mm)) + F_fuse + P_mem
```

`P_mem` 是可学习 2D position embedding。这个 memory 后续给 transformer decoder 查询。

本仓库当前状态：

- 当前只通过 `base_model.get_image_features(...)` 取 Qwen3-VL 视觉 token。
- 没有 patch Qwen3-VL visual forward 来取多层中间特征。
- 没有 `SpatialFeatureInjector`、多层融合和可学习 2D position embedding。
- 已用浅层 RGB 分支补了一部分高分辨率纹理，但不是论文的 multi-scale ViT injection。

在当前数据规模下，建议暂缓完整 multi-scale ViT patch。先做指标和 bbox 鲁棒性验证；如果确实卡在边界 P@0.9，再考虑加一个最小版 spatial injection。

### 2. Spatial-Semantic Query Construction

论文不只用语言 token 做 query，而是把 bbox 几何也编码进去。bbox 为：

```text
B_box = (x1, y1, x2, y2)
```

论文用 Fourier positional encoding 编码：

```text
E_box = gamma(x1) concat gamma(y1) concat gamma(0.2 * log(w) + 0.5) concat gamma(0.2 * log(h) + 0.5)
```

然后把 box embedding 和 mask token 语义相加：

```text
Q_seg^0 = LayerNorm(MLP_box(E_box) + W_seg(T_seg))
Q_seg^1 = TransformerDecoder(Q_seg^0, F_mem)
```

关键细节：

- bbox 不只是 gate，也参与 query 初始化。
- `log(w)` / `log(h)` 让尺度变化更稳定。
- `T_seg` 来自文本输出里的 mask placeholder，它把当前 mask decoder 和 JSON 中对应目标绑定起来。

本仓库当前状态：

- 当前没有使用 `T_seg`，因为 Phase 1/2B 下 base Qwen3-VL 在 `no_grad` 中运行，训练目标只有 mask loss。
- 当前没有 transformer decoder query。
- bbox 只作为二维 gate 进入 mask head。

建议后续最小可复现版本不要直接上完整 transformer decoder。更稳妥的增量是先加 `BoxEmbedding`：

```text
normalized bbox -> Fourier PE -> MLP -> hidden_dim
```

再用它对 mask feature 做 bias 或 FiLM：

```text
features = features * (1 + scale(box)) + bias(box)
```

这样能复现“bbox 参与语义/特征调制”的思想，但不需要从 Qwen 文本 hidden state 中抽 `T_seg`。

### 3. Box-Guided High-Resolution Pixel Fusion

这是当前仓库最值得保留和继续打磨的论文组件。论文用原图浅层 CNN 提取纹理：

```text
F_cnn = Stem(I)
```

直接融合浅层纹理会引入背景噪声，所以根据 bbox 构造 soft gate。论文把 bbox 宽高扩大 15%，再用 sigmoid 边界：

```text
M(x, y) =
  sigmoid(alpha * (x - x1'))
* sigmoid(alpha * (x2' - x))
* sigmoid(alpha * (y - y1'))
* sigmoid(alpha * (y2' - y))
```

论文 `alpha = 20`。多 bbox 时对各 gate 取空间最大值。

之后将高层视觉特征上采样，与 gated shallow feature 融合：

```text
F_pixel = F_up concat (M * F_cnn)
```

本仓库当前状态：

- `_box_gate(...)` 已实现 hard gate 和 soft gate。
- `seg_box_expand=0.15`、`seg_box_alpha=20.0` 与论文一致。
- `TongueMaskHead.image_stem` 已实现 RGB shallow branch。
- 当前实现是 `features * box_gate + image_stem(seg_images) * box_gate`，再 concat `box_gate` 走 conv head。

后续可以继续围绕这个模块优化，因为它最符合当前数据和硬件约束。比起大规模 LoRA 或 full fine-tune，优先级更高。

### 4. Iterative Mask-Aware Query Refinement

论文先预测一轮 mask，再用 soft mask 在 pixel feature 上做 target-aware pooling：

```text
M_logit^1 = Psi(Q_seg^1, F_pixel)
F_tar = sum(sigmoid(M_logit^1) * F_pixel) / (sum(sigmoid(M_logit^1)) + eps)
Q_seg^2 = LayerNorm(Q_seg^1 + phi_ref(F_tar))
M_logit^2 = Psi(Q_seg^2, F_pixel)
```

最后上采样到目标分辨率，并用 IoU head 输出 mask confidence：

```text
M_hat = Interp(M_logit^2)
s_iou = W_iou(Q_seg^2)
```

本仓库当前状态：

- Phase 2B 已有简化 refinement：第一轮 logits 经 sigmoid 后作为 `mask_hint`，拼到 feature 和 box gate 后走 `refine_net`。
- 当前没有 query pooling、dynamic kernel 和 IoU head。
- 当前 refinement 对 val/test 有明确提升，说明这个方向对舌体边界有效。

后续不要急着实现完整 dynamic mask head。当前更直接的增强是：

- 给评估加 P@0.9，专门观察高精度边界。
- 针对 `origin_Image/277.png` 这类大舌体欠分割样本看 error overlay。
- 如果确实是边界欠分割，再考虑 boundary loss 或 signed distance / edge-aware loss。

## 训练逻辑对照

论文训练分两阶段。

Stage 1 是 segmentation-centric adaptation：

- backbone：Qwen3-VL-4B
- LLM 原始权重冻结
- LoRA rank 32 可训练
- vision encoder 可训练
- mask decoder 可训练
- 训练 10,000 iterations
- 初始学习率 `1e-4`
- vision encoder 学习率为 `0.01x`
- loss = text generation loss + segmentation loss
- segmentation loss = BCE + Dice

Stage 2 是 perception/understanding synergy：

- 合并 Stage 1 LoRA 到 LLM backbone
- LLM backbone 和 mask decoder 全量微调
- vision encoder 冻结
- 训练 5,000 iterations
- 学习率 `7e-7`
- 数据混合比例为 referring segmentation : general multimodal understanding : multimodal reasoning = 3 : 1 : 2

本仓库当前训练逻辑：

- Qwen3-VL base 全冻结。
- `forward(...)` 中 base model 在 `no_grad` 下运行。
- 只训练 `mask_head.*`。
- 不计算有效 LM loss，`loss = seg_loss_weight * (BCE + Dice)`。
- checkpoint 只保存 `mask_head.*`，避免保存 4B base 权重。
- `data_flatten`、`data_packing`、MoE、LoRA 在分割模式下不支持。

这个取舍是合理的。当前只有 300 张单类舌体数据，直接按论文 Stage 1/2 训练会有三个问题：

- 数据规模远小于论文的 RefCOCO/LVIS/COCO/SA1B-ORS 混合数据。
- 24GiB 显存下训练 4B + vision encoder + LoRA 的余量有限。
- 单类任务不需要开放世界语言能力，训练 LLM/vision encoder 很容易过拟合或破坏原模型能力。

因此后续复现应保持“先做 decoder 逻辑，后做 trainable backbone”的顺序。

## 数据构造对照

论文 SA1B-ORS 有两个子集：

- SA1B-CoRS：category-oriented，表达式对应一个类别，可包含一个或多个实例。
- SA1B-DeRS：descriptive，表达式用属性、关系、上下文区分单个实例。

SA1B-CoRS  pipeline：

```text
SA-1B raw images
  -> RAM++ tag
  -> semantic filtering
  -> grounding verification
  -> Qwen3-VL-Plus bbox
  -> SAM2 coarse mask
  -> SA-1B fragment merging
  -> morphology / connected component cleanup
  -> MLLM verification
  -> caption generation
```

SA1B-DeRS pipeline：

```text
target instance + mask overlay + bbox
  -> Qwen3-VL-Plus generates descriptive instruction
  -> feed instruction back to Qwen3-VL-Plus for grounding
  -> keep sample only if predicted bbox IoU >= 0.8
  -> saliency filtering by mask area ratio
```

本仓库不能复现这条数据 pipeline。原因不是代码量，而是数据和外部模型依赖不匹配：

- 没有 SA-1B 级别图像池。
- 不需要 RAM++、SAM2、Qwen3-VL-Plus 级别的数据清洗。
- 舌体数据是单类、单实例、已有 GT mask，不需要开放世界类别蒸馏。

当前本地数据等价于论文中的一个极简 supervised source：

```text
image + GT mask
  -> bbox from mask
  -> label = tongue
  -> fixed category instruction
  -> JSON output with mask placeholder
```

后续可做的数据增强应围绕当前任务，而不是仿 SA1B-ORS：

- 保持固定 test split，不用 test 调参。
- 增加 train-only 几何和颜色增强时，mask 与 bbox 必须同步更新。
- 增加 bbox jitter，模拟推理阶段 bbox 不准的情况。
- 加一个 generated-bbox 评估，区分 mask head 上限和端到端能力。

## 当前最大复现缺口

当前指标高，但仍有一个关键缺口：评估主要使用 GT bbox。

论文真正的端到端路径是：

```text
Qwen3-VL predicts bbox -> decoder uses predicted bbox -> mask
```

当前主要路径是：

```text
GT mask computes bbox -> decoder uses GT bbox -> mask
```

因此当前 Phase 2B 指标更像是“给定准确 bbox 后，mask decoder 能否切出舌体”。这对验证 decoder 有价值，但还不能说明 Qwen3-VL-Seg 式端到端复现完成。

后续必须拆成三套评估：

| 评估模式 | bbox 来源 | 目的 |
| --- | --- | --- |
| upper-bound | GT bbox | 验证 mask head 上限 |
| robustness | GT bbox + jitter | 验证 bbox 误差容忍度 |
| end-to-end | Qwen3-VL generated bbox | 验证真实推理链路 |

如果 generated bbox 的 IoU 不稳定，继续堆 mask decoder 不会解决端到端问题。要先量化 bbox 质量。

## 后续复现路线

### R0. 固化当前最佳基线

目标是确保任何后续改动都有可比较基准。

保留：

- `outputs/tongue_seg_phase2b/model.safetensors`
- `outputs/tongue_seg_phase2b/run_config.json`
- `outputs/tongue_seg_phase2b/train_log.jsonl`
- val/test 的 `summary.json`、`metrics.xlsx`、`predictions.jsonl`、overview/overlay

推荐继续以当前命令作为基线：

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

验收：

- val/test Dice 和 mIoU 与 README 中结果接近。
- checkpoint 只包含 `mask_head.*`。
- `train_log.jsonl` 有 `seg_loss`、`bce_loss`、`dice_loss`、`pred_area_ratio`、`gt_area_ratio`。

### R1. 补论文评估指标

论文使用 mIoU、cIoU、P@0.5 / P@0.7 / P@0.9。当前 README 主要报告 Dice/mIoU。

建议先改 `eval_tongue_seg.py`：

- 增加 cIoU：
  ```text
  sum(intersection over dataset) / sum(union over dataset)
  ```
- 增加 P@0.5、P@0.7、P@0.9：
  ```text
  mean(sample_iou >= threshold)
  ```
- 保留 Dice，因为医学/舌体分割常用。
- 可选增加 boundary IoU 或 Boundary F-score，但不要先加复杂后处理。

为什么优先做 P@0.9：

- 论文消融显示高分辨率图像分支和多尺度特征主要提升 P@0.9。
- 当前平均 Dice 已经很高，普通 mIoU 可能掩盖边界问题。
- `origin_Image/277.png` 这类欠分割样本需要更敏感的边界指标。

验收：

- `summary.json` 同时包含 Dice、mIoU、cIoU、P@0.5、P@0.7、P@0.9。
- `metrics.xlsx` 每样本包含 IoU、Dice、面积比例、文件名、bbox。

### R2. 增加 bbox 鲁棒性评估

目标是补上当前 GT bbox 评估和论文 predicted bbox 推理之间的缺口。

建议在评估脚本中加入可选参数：

```text
--bbox_jitter 0.00
--bbox_jitter 0.05
--bbox_jitter 0.10
--bbox_jitter 0.15
```

jitter 只用于 eval，不改 JSON：

- 中心点随机平移
- 宽高随机缩放
- clamp 到 `[0, 1]`
- 记录 jitter 后 bbox 与 GT bbox 的 IoU

输出对比：

| bbox_jitter | Dice mean | mIoU mean | P@0.9 | bbox IoU mean |
| --- | ---: | ---: | ---: | ---: |
| 0.00 | 当前上限 | 当前上限 | 当前上限 | 1.0 |
| 0.05 | 待测 | 待测 | 待测 | 待测 |
| 0.10 | 待测 | 待测 | 待测 | 待测 |

如果 `0.05` jitter 就明显掉点，说明当前 mask head 依赖精确 bbox，后续应优先增强 soft gate 和 bbox 外扩，而不是训练 LoRA。

### R3. 增加 generated bbox 端到端评估

目标是验证当前模型是否接近论文的真实推理路径。

流程：

```text
image + prompt
  -> base Qwen3-VL generate JSON
  -> parse bbox_2d
  -> normalize / clamp bbox
  -> mask head predict mask
  -> 同时评估 bbox IoU 和 mask IoU
```

需要记录：

- `bbox_source = generated`
- 生成文本原文
- 是否成功解析 JSON/bbox
- generated bbox 与 GT bbox 的 IoU
- generated bbox 下的 Dice/mIoU/P@0.9

建议 prompt 保持与训练 JSON 一致：

```text
<image>
Locate and segment the tongue, report bbox coordinates and mask in JSON format.
```

如果 Qwen3-VL 生成 bbox 格式不稳定，先不要改模型。先做解析失败统计，并保存失败文本。

验收：

- 至少在 val/test 上跑通 generated-bbox 评估。
- 能回答两个问题：
  - base Qwen3-VL 对舌体 bbox 的平均 IoU 是多少？
  - mask 质量下降主要来自 bbox 不准，还是 mask head 本身？

### R4. 最小复现 spatial-semantic query

在 R1-R3 完成前，不建议上完整 transformer decoder。完成后如果仍想更贴近论文，可做一个低风险版本。

新增 `BoxEmbedding`：

```text
bbox -> Fourier PE -> MLP -> hidden_dim
```

接入方式：

```text
box_embed = MLP(Fourier(bbox))
scale, bias = Linear(box_embed).chunk(2)
features = features * (1 + scale[..., None, None]) + bias[..., None, None]
```

优点：

- 复现论文“bbox 参与 query/feature 调制”的关键思想。
- 不需要改 Qwen3-VL 语言 forward。
- 不需要提取 `<mask_token>` hidden state。
- 参数少，适合 300 张数据。

不要一开始做：

- 多层 transformer decoder
- dynamic kernel mask head
- 多实例 query matching
- IoU head 参与训练

验收：

- 参数量小幅增加。
- val/test P@0.9 或最差样本改善。
- Dice/mIoU 不低于 Phase 2B。

### R5. 最小复现 multi-scale spatial injection

论文的 multi-scale ViT injection 需要从 Qwen3-VL vision encoder 中拿中间层特征。当前 transformers 接口未必直接暴露，贸然 patch 风险较高。

建议分两步：

第一步不 patch Qwen：

- 用 `seg_images` 建一个轻量 image pyramid stem。
- 例如 `256x256`、`128x128`、`64x64` 三个尺度。
- 与当前 Qwen visual token feature 融合。

第二步再考虑 patch Qwen visual forward：

- 确认 Qwen3-VL visual encoder 每层输出 shape。
- 只取 2-3 个中间层，不取全层。
- 用论文的 near-identity adapter：
  ```text
  x + s * depthwise_conv_norm_gelu(x), s init 1e-3
  ```
- 先在 2B 上 smoke，再上 4B。

验收：

- 不引入明显 OOM。
- 10 step smoke 正常。
- P@0.9 或 worst-case overlay 有改善。

### R6. 谨慎尝试 LoRA / vision tuning

论文 Stage 1 训练 LoRA、vision encoder 和 mask decoder。但本仓库不应直接照搬。

只有满足下面条件再做：

- R1-R3 已确认端到端瓶颈不是 bbox 解析/定位。
- R4/R5 的轻量 decoder 改造收益有限。
- 训练/评估脚本已经能稳定记录每个 variant 的结果。

建议顺序：

1. 2B + LoRA smoke，确认保存和加载。
2. 4B + LoRA only，不解冻 vision encoder。
3. 如果必须解冻 vision encoder，只用极小 lr，并先做短训。

风险：

- 300 张数据很容易过拟合。
- 4B + vision encoder 训练可能接近 24GiB 显存边界。
- 当前任务是单类分割，LoRA 对 mask 边界未必有收益。
- 一旦训练 LLM，checkpoint 保存逻辑要重新设计，不能再只保存 `mask_head.*`。

## 不建议当前复现的内容

以下内容属于论文完整系统的一部分，但不适合当前仓库阶段：

- SA1B-ORS 数据构造全流程。
- ORS-Bench 构造。
- 多实例 category segmentation。
- description-based open-world referring segmentation。
- Stage 2 的 3:1:2 混合训练。
- 全量 LLM fine-tuning。
- MoE Qwen3-VL-Seg。
- 完整 17M decoder 的逐层复刻，除非论文代码公开或先完成端到端 bbox 评估。

这些内容不是“不重要”，而是当前数据、硬件和任务目标不支持可靠验证。

## 推荐的下一步优先级

优先级 1：把评估补完整。

- cIoU
- P@0.5 / P@0.7 / P@0.9
- worst-case overlay
- bbox IoU 字段

优先级 2：把 bbox 来源拆开。

- GT bbox upper bound
- jitter bbox robustness
- generated bbox end-to-end

优先级 3：针对边界问题做轻量改造。

- bbox Fourier embedding / FiLM
- bbox jitter training
- boundary loss 或 edge-aware loss
- image pyramid stem

优先级 4：最后再考虑 LoRA。

- 先 2B smoke
- 再 4B LoRA only
- vision encoder tuning 暂缓

## 最小验收标准

后续每个复现实验都应至少记录：

- git commit
- model path
- checkpoint path
- train split / val split / test split
- bbox 来源
- seg 参数：`mask_size`、`box_expand`、`box_alpha`、`highres_fusion`、`refine`
- Dice mean
- mIoU mean
- cIoU
- P@0.5 / P@0.7 / P@0.9
- pred/GT 面积比例
- worst 5 overlays
- 是否使用 test 调参

一个改动只有在下面条件同时满足时才算有效：

- val/test 均不低于 Phase 2B 主指标。
- P@0.9 或 worst-case overlay 有改善。
- generated-bbox 模式没有明显退化。
- checkpoint 体积和保存逻辑符合当前仓库策略。
- 代码改动没有把简单单类任务扩展成不可维护的大框架。

## 结论

当前仓库已经实现了论文中最适合本任务的三个核心偏置：

- bbox 作为结构先验
- box-guided high-resolution fusion
- mask-aware refinement

下一阶段不应优先追求“看起来更像论文”的大 decoder，而应先补齐论文式评估指标和端到端 bbox 评估。只有确认瓶颈确实在 mask decoder 边界表达后，再逐步加入 bbox Fourier embedding、轻量多尺度注入或更复杂的 query refinement。

