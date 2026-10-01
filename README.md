# SLM — 字符级 GPT 训练推理流水线

基于 PyTorch 实现的微型 GPT（Decoder-only Transformer），从零训练字符级语言模型，适合学习 Transformer 原理及中文对话语料微调实验。

## 项目结构

```
.
├── model.py          # 公共模型定义（GPT / TransformerBlock / Attention / MLP）
├── preprocess.py     # 语料预处理：parquet 对话文件 → 纯文本 txt
├── train.py          # 训练脚本：数据加载 → 训练 → 评估 → 可视化 → 导出
├── inference.py      # 推理脚本：交互式生成 / 命令行单次生成
└── README.md
```

## 依赖环境

| 包 | 用途 |
|---|------|
| `torch >= 2.0` | 核心框架（推荐 CUDA 版本） |
| `matplotlib` | 损失曲线可视化 |
| `pandas` | 数据加载与损失表格导出 |
| `pyarrow` | parquet 文件读取（仅 preprocess.py） |
| `tqdm` | 进度条（仅 preprocess.py） |

安装：

```bash
pip install torch matplotlib pandas pyarrow tqdm
```

## 快速开始

### 1. 预处理语料

将 parquet 格式的对话语料放入目标文件夹，修改 `preprocess.py` 中的路径后运行：

```bash
python preprocess.py
```

产物：`input/` 目录下生成编号 `.txt` 文件。

支持的 parquet 列结构：
- `conversations` — 多轮对话（自动格式化为 `<|role|>: content`）
- `text` — 纯文本
- `source` + `answer` — 问答对（格式化为 `问题: ...\n答案: ...`）

### 2. 训练

```bash
python train.py
```

产物：

| 文件 | 说明 |
|------|------|
| `mini_gpt.pt` | 最终模型权重 |
| `mini_gpt_best.pt` | 验证集损失最低的模型 |
| `checkpoint.pt` | 训练检查点，支持中断恢复 |
| `vocab.json` | 字符映射表 + 超参数（推理自动读取） |
| `loss_curve.png` | 训练/验证损失曲线 |
| `loss_history.csv` | 损失数据表 |
| `training_log.txt` | 训练配置与结果日志 |
| `generated_text.txt` | 训练结束后的生成样本 |

#### 超参数说明

所有超参数在 `train.py` 顶部集中配置，修改即生效：

```python
block_size = 256      # 上下文窗口长度
batch_size = 16       # 批次大小
n_layer = 6           # Transformer 层数
n_head = 8            # 注意力头数
n_embd = 256          # 嵌入维度
dropout = 0.08        # Dropout 概率

learning_rate = 3e-4  # 峰值学习率
min_lr = 1e-5         # 余弦退火最低学习率
weight_decay = 0.01   # 权重衰减
grad_clip = 1.0       # 梯度裁剪阈值

max_iters = 50000     # 总训练步数
warmup_iters = 1000   # 学习率预热步数
max_chars = 5_000_000 # 最大加载字符数（防止 OOM）
```

#### 中断恢复

训练脚本每次评估（每 2000 步）自动保存 `checkpoint.pt`，中断后直接重新运行 `python train.py` 即可从上次断点继续。

#### 训练特性

| 特性 | 说明 |
|------|------|
| 余弦学习率调度 | 线性预热 → 余弦退火至 min_lr |
| AdamW + 权重衰减 | 防止过拟合 |
| 梯度裁剪 | 防止 loss 爆炸 |
| 混合精度 (AMP) | CUDA 下自动启用，加速训练 |
| torch.compile | PyTorch ≥ 2.0 自动启用 |
| 最佳模型保存 | 每次评估后按验证集 loss 更新 |

### 3. 推理

```bash
# 交互模式（推荐调试用）
python inference.py

# 单次生成
python inference.py "你好，今天天气怎么样？"

# 指定参数
python inference.py --prompt "请介绍一下自己" --tokens 300 --temp 0.7
```

交互模式支持实时调参：

| 命令 | 说明 |
|------|------|
| `:temp 0.7` | 调整温度（低→确定性，高→随机性） |
| `:topk 50` | 调整 top-k 采样 |
| `:tokens 300` | 调整生成长度 |
| `q` / `quit` | 退出 |

生成参数说明：

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `temperature` | 控制随机性：0=贪婪，接近1=随机，>1=更随机 | 0.7 ~ 0.9 |
| `top_k` | 只从概率最高的 k 个字符中采样 | 30 ~ 50 |
| `max_tokens` | 最大生成的字符数 | 100 ~ 500 |

## 模型架构

```
GPT (Decoder-only Transformer)
├── Token Embedding   (vocab_size × 256)
├── Position Embedding (block_size × 256)
├── Dropout
├── TransformerBlock × 6 (Pre-LN)
│   ├── LayerNorm → CausalSelfAttention (8 heads)
│   └── LayerNorm → MLP (256 → 1024 → 256, GELU)
├── LayerNorm (final)
└── Linear Head → vocab_size  (weight tying with embedding)
```

约 **3.5M** 参数量（以默认配置计）。

## 常见问题

**Q: 训练 loss 一直很高，几乎不降？**

检查语料量是否足够。字符级模型对数据量敏感，建议至少 1M 字符（约 50 万字中文）。如果语料不足，可以降低 `max_iters` 并用现有数据微调。

**Q: 生成结果是乱码？**

可能原因：
1. `vocab.json` 与模型不匹配 — 用同一次训练生成的 `vocab.json`
2. 超参数不一致 — 推理从 `vocab.json` 的 `hyperparams` 字段自动读取，正常情况下不应出现此问题
3. 训练不充分 — 检查 loss 曲线是否收敛

**Q: OOM（显存不足）？**

调整 `train.py`：
- 降低 `block_size`（128）
- 降低 `batch_size`（8）
- 降低 `max_chars`
- 降低 `n_layer` 或 `n_embd`

**Q: 预处理报错 "No such file or directory"？**

`preprocess.py` 中的 `CORPUS_FOLDER` 路径需修改为你的实际 parquet 文件夹路径。如果 parquet 文件在子文件夹中，脚本会递归搜索，无需担心。

**Q: 如何用自己的 txt 语料直接训练？**

跳过 preprocess，直接把 txt 文件放入 `input/` 文件夹（与 train.py 同目录），然后运行 `python train.py`。