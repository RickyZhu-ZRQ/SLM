"""
导出训练好的 PyTorch 模型权重为纯二进制格式，供 C++ 推理使用。

用法: python export_weights.py [model_path] [vocab_path] [output_dir]
"""

import sys
import os
import json
import struct
import gc
import torch

# 添加当前目录到 path，确保能 import model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT


def export(model_path, vocab_path, output_dir):
    device = torch.device("cpu")

    # 加载 vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    stoi = vocab_data["stoi"]
    itos = {int(k): v for k, v in vocab_data["itos"].items()}
    vocab_size = vocab_data["vocab_size"]

    # 提取超参数（优先从 vocab.json，否则用默认值）
    if "hyperparams" in vocab_data:
        hp = vocab_data["hyperparams"]
        n_embd = hp["n_embd"]
        n_layer = hp["n_layer"]
        n_head = hp["n_head"]
        block_size = hp["block_size"]
        dropout = hp.get("dropout", 0.08)
    else:
        n_embd, n_layer, n_head, block_size, dropout = 256, 6, 8, 256, 0.08

    print(f"超参数: n_embd={n_embd}, n_layer={n_layer}, n_head={n_head}, "
          f"block_size={block_size}, vocab_size={vocab_size}")

    # 加载模型
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model = GPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        dropout=dropout,
    )
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # ========== 导出 vocab（字符映射表） ==========
    vocab_txt = os.path.join(output_dir, "vocab.bin")
    with open(vocab_txt, "w", encoding="utf-8") as f_v:
        f_v.write(f"vocab_size {vocab_size}\n")
        for i, ch in sorted(itos.items()):
            f_v.write(f"{i} {ch}\n")
    print(f"vocab 已导出: {vocab_txt}")

    # ========== 导出权重为单个二进制文件（流式写入，避免 tensors 列表积累） ==========
    weights_path = os.path.join(output_dir, "model_weights.bin")

    tensor_names = list(state_dict.keys())
    n_tensors = len(tensor_names)

    with open(weights_path, "wb") as f:
        # Header: magic + config
        header = struct.pack(
            "4s 6i",
            b"KANG",
            n_embd, n_layer, n_head, block_size, vocab_size, n_tensors,
        )
        f.write(header)

        total_params = 0
        for name in tensor_names:
            data = state_dict[name].detach().cpu().to(torch.float32).numpy()
            # 写入 name
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("I", len(name_bytes)))
            f.write(name_bytes)
            # 写入维度
            shape = data.shape
            f.write(struct.pack("I", len(shape)))
            for s in shape:
                f.write(struct.pack("I", s))
            # 写入数据
            f.write(data.tobytes())
            total_params += data.size
            # 立即释放 numpy 数组
            del data

    print(f"权重已导出: {weights_path}")
    print(f"  张量数: {n_tensors}, 总参数: {total_params:,}")
    for name in tensor_names[:5]:
        val = state_dict[name]
        print(f"  {name}: {list(val.shape)}")
    if n_tensors > 5:
        print(f"  ... 共 {n_tensors} 个张量")


# model 不再需要，释放它持有的两份权重复制（model 自身 + ckpt ）
    del model, ckpt, state_dict
    gc.collect()


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "mini_gpt.pt"
    vocab_path = sys.argv[2] if len(sys.argv) > 2 else "vocab.json"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "cpp"
    export(model_path, vocab_path, output_dir)