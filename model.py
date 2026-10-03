"""
共享模型定义：GPT 字符级语言模型
供 train.py / inference.py 统一导入，修改超参数只需改一处。
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 全局优化开关（训练脚本启动时设置一次即可） ----
_OPTIMIZED = False


def enable_optimizations():
    """在训练脚本开头调用一次，全局生效。"""
    global _OPTIMIZED
    if _OPTIMIZED:
        return
    _OPTIMIZED = True
    # ① 允许 PyTorch 用 TF32 精度做矩阵乘（CPU 上也有效果）
    torch.set_float32_matmul_precision("high")

    # ② 如果支持 bfloat16 AMP（CPU），后续训练循环可用
    #    检测写在 train.py 里，这里只提供开关


# ---- 优化版因果自注意力（SDPA） ----
class CausalSelfAttention(nn.Module):
    """因果自注意力层 — 使用 PyTorch SDPA 后端自动选择最优核"""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key   = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj  = nn.Linear(n_embd, n_embd)
        self.attn_dropout_p = dropout  # SDPA 不支持 per-forward dropout 参数
        self.resid_dropout  = nn.Dropout(dropout)
        # 保留 bias buffer 供旧版兼容（SDPA 路由走 is_causal=True，不读它）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        # Q/K/V: (B, T, C) → (B, n_head, T, head_dim)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # SDPA：PyTorch 自动选择最优后端（CPU 上用 math/aten 内核，GPU 上用 FlashAttention）
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )
        # y: (B, n_head, T, head_dim) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.resid_dropout(y)
        return y


# ---- MLP（不变） ----
class MLP(nn.Module):
    """两层全连接前馈网络（GELU 激活）"""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ---- Transformer Block（不变） ----
class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block"""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---- GPT（微调：前向允许指定内存格式） ----
class GPT(nn.Module):
    """字符级 GPT 语言模型"""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        block_size: int = 256,
        dropout: float = 0.08,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # weight tying: embedding 与 lm_head 共享权重
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        assert T <= self.block_size, f"输入长度 {T} 超过 block_size {self.block_size}"
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # fused cross-entropy: view inside C is more efficient
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """自回归生成"""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train(was_training)
        return idx


def get_device() -> str:
    """自动选择最佳设备"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"