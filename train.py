"""
字符级 GPT 训练脚本
用法: python train.py
"""

import csv
import gc
import math
import os
import re
import sys
import glob
import json
import time
import unicodedata
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 所有路径基于脚本所在目录，不依赖 CWD
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from model import GPT, get_device, enable_optimizations

# ============================================================
# 全局优化开关
# ============================================================
enable_optimizations()

# CPU 线程配置：interop=1 对单 GPU/CPU 训练显著更快（避免线程竞争）
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(1)

# ============================================================
# 文本清洗：移除噪声字符
# ============================================================
# 控制字符（保留 tab / LF / CR 用于文本结构）
_RE_CONTROL = re.compile(
    "[" "\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f" "]", flags=re.UNICODE
)

def clean_text(text: str) -> str:
    """移除不可渲染/干扰字符，保留正常 Unicode 文本"""
    # 1. C0/C1 控制字符（除 tab/LF/CR）
    text = _RE_CONTROL.sub("", text)
    # 2. 按行处理：过滤行内不可见字符
    lines = []
    for line in text.split("\n"):
        cleaned_chars = []
        for ch in line:
            cp = ord(ch)
            # 跳过私用区 (PUA: BMP + 平面15 + 平面16)
            if (0xE000 <= cp <= 0xF8FF) or (0xF0000 <= cp <= 0x10FFFD):
                continue
            # 跳过非字符 (noncharacters)
            if (cp & 0xFFFE) == 0xFFFE and (0xFDD0 <= cp <= 0xFDEF or cp >= 0xFFFE):
                continue
            # 跳过零宽/格式字符 + 未分配码点
            cat = unicodedata.category(ch)
            if cat in ("Cf", "Cn"):  # Format chars, Unassigned
                continue
            cleaned_chars.append(ch)
        lines.append("".join(cleaned_chars))
    return "\n".join(lines)

# ============================================================
# 超参数
# ============================================================
block_size = 256         # 上下文长度
batch_size = 8           # 微批次大小（单次前向的 batch）
grad_accum_steps = 2      # 梯度累积步数 → 有效 batch = 8×2 = 16
n_layer = 6              # Transformer 层数
n_head = 8               # 注意力头数
n_embd = 256             # 嵌入维度
dropout = 0.08           # Dropout 概率

# 优化器
learning_rate = 3e-4     # 峰值学习率
min_lr = 1e-5            # 余弦调度的最低学习率
weight_decay = 0.01      # AdamW 权重衰减
grad_clip = 1.0          # 梯度裁剪阈值

# 训练控制
max_iters = 50000       # 总训练步数
warmup_iters = 1000      # 学习率预热步数
eval_interval = 999999   # 不评估（设得比 max_iters 大即永远不触发）
eval_iters = 200         # 每次评估的 batch 数
log_interval = 50        # 进度汇报间隔（仅步数）
detail_interval = 1000   # 详细日志间隔（loss/lr/耗时）

# 数据
input_folder = os.path.join(_SCRIPT_DIR, "input")   # 预处理生成的 txt 文件夹
max_chars = 1_000_000_000   # 最大加载字符数

# ============================================================
# 设备与混合精度
# ============================================================
device = get_device()

# 仅 CUDA 使用 AMP（MPS 的 AMP 不稳定）
use_amp = (device == "cuda")
if device == "cuda":
    torch.backends.cudnn.benchmark = True

print(f"设备: {device}, 混合精度: {use_amp}")


# ============================================================
# 数据加载
# ============================================================
def load_text(folder: str, max_chars: int) -> str:
    """从文件夹中加载 txt 文件并拼接"""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"找不到文件夹 {folder}，请先运行 preprocess.py")

    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not files:
        raise FileNotFoundError(f"文件夹 {folder} 中没有 txt 文件")

    print(f"找到 {len(files)} 个 txt 文件，开始读取...")
    text = ""
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            text += f.read()
        if len(text) >= max_chars:
            text = text[:max_chars]
            print(f"已达到最大字符数限制 {max_chars:,}")
            break
    print(f"加载文本成功，总字符数: {len(text):,}")
    return text


text = load_text(input_folder, max_chars)

# 清洗噪声字符（控制字符、零宽字符、私用区码点等）
before = len(text)
text = clean_text(text)
removed = before - len(text)
if removed > 0:
    print(f"清洗噪声字符: {removed:,} 个 ({100*removed/before:.2f}%) 被移除")

# 字符映射
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

print(f"词汇表大小: {vocab_size}")

# 保存 vocab.json（附带超参数，供推理脚本自动读取）
with open(os.path.join(_SCRIPT_DIR, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump({
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "vocab_size": vocab_size,
        "hyperparams": {
            "n_embd": n_embd,
            "n_layer": n_layer,
            "n_head": n_head,
            "block_size": block_size,
            "dropout": dropout,
        },
    }, f, ensure_ascii=False)
print("字符映射已保存为 vocab.json（含超参数）")

# 数据分割 — text 编码后释放原始字符串
data = torch.tensor(encode(text), dtype=torch.int32)
n_train = int(0.9 * len(data))
train_data = data[:n_train]
val_data = data[n_train:]
del text  # 释放 5M+ 字符串内存
gc.collect()

# 数据量校验
min_required = block_size + batch_size * 2
if len(train_data) <= block_size or len(val_data) <= block_size:
    raise RuntimeError(
        f"数据量不足: train={len(train_data)}, val={len(val_data)}, "
        f"block_size={block_size}。需要至少 {block_size + 1} 个 token。"
        f"\n  当前 max_chars={max_chars}，建议增大该值或增加语料。"
    )


def get_batch(split: str):
    """随机采样一个 batch"""
    buf = train_data if split == "train" else val_data
    ix = torch.randint(len(buf) - block_size, (batch_size,))
    x = torch.stack([buf[i : i + block_size] for i in ix])
    y = torch.stack([buf[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device).long(), y.to(device).long()


# ============================================================
# 模型构建
# ============================================================
model = GPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head,
    block_size=block_size,
    dropout=dropout,
).to(device)

# torch.compile: 需要 C++ 编译器，当前环境不可用 → 跳过
# 改用 set_float32_matmul_precision("high") + SDPA (已在 enable_optimizations() 中设置)
print("优化: SDPA + fp32_matmul_precision=high")

# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate,
    weight_decay=weight_decay, betas=(0.9, 0.95),
)

# 混合精度
scaler = torch.cuda.amp.GradScaler() if use_amp else None

# 训练状态恢复
start_iter = 0
checkpoint_path = os.path.join(_SCRIPT_DIR, "checkpoint.pt")
if os.path.exists(checkpoint_path):
    print(f"发现检查点 {checkpoint_path}，正在恢复...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_iter = ckpt["iter"] + 1
    print(f"从 step {start_iter} 继续训练")

# 学习率调度器
def get_lr(iter: int) -> float:
    """余弦退火 + 线性预热"""
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    if iter > max_iters:
        return min_lr
    decay_ratio = (iter - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + (learning_rate - min_lr) * (1 + math.cos(math.pi * decay_ratio)) / 2


# ============================================================
# 训练
# ============================================================
total_params = sum(p.numel() for p in model.parameters()) / 1e6
effective_batch = batch_size * grad_accum_steps
print(f"\n{'='*50}")
print(f"模型参数量: {total_params:.3f}M")
print(f"词汇表大小: {vocab_size}")
print(f"训练步数: {start_iter} → {max_iters}")
print(f"微批次={batch_size} × 累积={grad_accum_steps} → 有效批次={effective_batch}")
print(f"block_size={block_size}")
print(f"n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")
print(f"lr={learning_rate}, min_lr={min_lr}, warmup={warmup_iters}")
print(f"{'='*50}\n")

train_losses, val_losses, eval_steps = [], [], []
train_ppls, val_ppls = [], []


@torch.no_grad()
def estimate_loss():
    """评估当前模型在 train/val 上的平均 loss"""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


start_time = time.time()
best_val_loss = float("inf")

for iter in range(start_iter, max_iters):
    iter_start = time.time()

    # --- 学习率更新 ---
    current_lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    # --- 评估（已禁用）---
    if False:
        losses = estimate_loss()
        # CUDA: 评估完后清一波碎片
        if device == "cuda":
            torch.cuda.empty_cache()
        eval_time = time.time() - iter_start
        train_ppl = math.exp(losses["train"])
        val_ppl = math.exp(losses["val"])
        print(
            f"\n[评估] step {iter:6d}: "
            f"train loss {losses['train']:.4f} (ppl {train_ppl:.2f}), "
            f"val loss {losses['val']:.4f} (ppl {val_ppl:.2f}), "
            f"耗时 {eval_time:.2f}s"
        )
        eval_steps.append(iter)
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        # 保存最佳模型
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), os.path.join(_SCRIPT_DIR, "mini_gpt_best.pt"))
            print(f"  -> 最佳模型已保存 (val_loss={best_val_loss:.4f})")

    # --- 训练一步（梯度累积） ---
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    for micro_step in range(grad_accum_steps):
        xb, yb = get_batch("train")

        if scaler is not None:
            with torch.cuda.amp.autocast():
                _, loss = model(xb, yb)
            # 用 loss/grad_accum_steps 累加梯度（总 loss 不变但梯度正确）
            scaler.scale(loss / grad_accum_steps).backward()
        else:
            _, loss = model(xb, yb)
            (loss / grad_accum_steps).backward()
        accum_loss += loss.item()

    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    avg_loss = accum_loss / grad_accum_steps

    # --- 进度条（同行动态更新） ---
    if iter % log_interval == 0:
        pct = iter / max_iters
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        elapsed = time.time() - start_time
        eta = elapsed / pct * (1 - pct) if pct > 0 else 0
        print(
            f"\r[训练] {bar} {iter}/{max_iters} ({pct*100:.1f}%) | "
            f"用时 {elapsed:.0f}s | 剩余 {eta:.0f}s",
            end="", flush=True,
        )

    # --- 详细日志 ---
    if iter % detail_interval == 0 or iter == max_iters - 1:
        print()  # 进度条换行
        step_time = time.time() - iter_start
        loss_val = avg_loss
        eff_tok_s = effective_batch * block_size / step_time
        mem_info = ""
        if device == "cuda":
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            mem_info = f" | 显存 {alloc:.1f}MB (预留 {reserved:.1f}MB)"
        print(
            f"[详细] step {iter:6d}/{max_iters} | "
            f"loss {loss_val:.4f} | lr {current_lr:.6f} | "
            f"耗时 {step_time:.3f}s | {eff_tok_s:.0f} tok/s{mem_info}"
        )

    # --- 定时保存检查点 ---
    if iter % 500 == 0 and iter > 0:
        torch.save({
            "iter": iter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

total_time = time.time() - start_time
print(f"\n训练完成，总耗时 {total_time:.2f}s ({total_time/3600:.2f}h)")

# 保存最终模型
torch.save(model.state_dict(), os.path.join(_SCRIPT_DIR, "mini_gpt.pt"))
torch.save({
    "iter": max_iters - 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, checkpoint_path)
print("模型已保存为 mini_gpt.pt，检查点已保存为 checkpoint.pt")


# ============================================================
# 损失曲线可视化
# ============================================================
fig, ax1 = plt.subplots(figsize=(12, 7))

ax1.plot(eval_steps, train_losses, label="Train Loss",
         color="tab:blue", marker="o", markersize=4, linewidth=2)
ax1.plot(eval_steps, val_losses, label="Validation Loss",
         color="tab:red", marker="s", markersize=4, linewidth=2)

# 标注（隔点标注避免过于密集）
for i, (x, y) in enumerate(zip(eval_steps, train_losses)):
    if i % max(1, len(eval_steps) // 10) == 0:
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color="tab:blue")
for i, (x, y) in enumerate(zip(eval_steps, val_losses)):
    if i % max(1, len(eval_steps) // 10) == 0:
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, -15), ha="center", fontsize=8, color="tab:red")

# 随机基线
random_loss = math.log(vocab_size)
ax1.axhline(y=random_loss, color="gray", linestyle="--", linewidth=1,
            label=f"Random baseline (ln({vocab_size})={random_loss:.2f})")

ax1.set_xlabel("Training Step")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Training and Validation Loss with Perplexity")
ax1.grid(True, which="both", linestyle=":", linewidth=0.5)
ax1.legend(loc="upper right")

# PPL 右轴
ax2 = ax1.twinx()
ax2.set_ylabel("Perplexity (exp(loss))")
ax2.set_yscale("log")
ax2.plot(eval_steps, train_ppls, color="tab:blue", linestyle="--", alpha=0.4,
         label="Train PPL (right axis)")
ax2.plot(eval_steps, val_ppls, color="tab:red", linestyle="--", alpha=0.4,
         label="Val PPL (right axis)")
ax2.legend(loc="center right")

fig.tight_layout()
fig.savefig(os.path.join(_SCRIPT_DIR, "loss_curve.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("损失曲线已保存为 loss_curve.png")


# ============================================================
# 生成示例
# ============================================================
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
generated_text = decode(generated[0].tolist())
print("\n=== 生成示例 ===")
print(generated_text)

# 保存训练数据
with open(os.path.join(_SCRIPT_DIR, "loss_history.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["step", "train_loss", "val_loss", "train_perplexity", "val_perplexity"])
    for row in zip(eval_steps, train_losses, val_losses, train_ppls, val_ppls):
        w.writerow(row)

with open(os.path.join(_SCRIPT_DIR, "training_log.txt"), "w", encoding="utf-8") as f:
    f.write(f"设备: {device}\n")
    f.write(f"模型参数量: {total_params:.3f}M\n")
    f.write(f"词汇表大小: {vocab_size}\n")
    f.write(f"训练步数: {max_iters}\n")
    f.write(f"batch_size={batch_size}, block_size={block_size}\n")
    f.write(f"n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}\n")
    f.write(f"lr={learning_rate}, min_lr={min_lr}, weight_decay={weight_decay}\n")
    f.write(f"训练总耗时: {total_time:.2f}s\n\n")
    f.write("====== 损失记录 ======\n")
    for step, tr, vl, tp, vp in zip(eval_steps, train_losses, val_losses, train_ppls, val_ppls):
        f.write(f"Step {step}: train_loss={tr:.4f} val_loss={vl:.4f} "
                f"train_ppl={tp:.2f} val_ppl={vp:.2f}\n")
    f.write("\n====== 生成示例 ======\n")
    f.write(generated_text)

with open(os.path.join(_SCRIPT_DIR, "generated_text.txt"), "w", encoding="utf-8") as f:
    f.write(generated_text)

print("损失历史 → loss_history.csv")
print("训练日志 → training_log.txt")
print("生成文本 → generated_text.txt")