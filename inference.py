"""
字符级 GPT 推理脚本
用法:
    python inference.py                          # 交互模式
    python inference.py "你好"                   # 单次生成
    python inference.py --prompt "你好" --tokens 300 --temp 0.7
"""

import os
import sys
import json
import argparse
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from model import GPT, get_device

device = get_device()


def load_vocab_and_hp(vocab_path: str):
    """加载 vocab.json 并提取超参数"""
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stoi = data["stoi"]
    itos = {int(k): v for k, v in data["itos"].items()}
    vocab_size = data["vocab_size"]

    if "hyperparams" in data:
        hp = data["hyperparams"]
    else:
        hp = {"n_embd": 256, "n_layer": 6, "n_head": 8, "block_size": 256, "dropout": 0.08}

    encode_fn = lambda s: [stoi[c] for c in s]
    decode_fn = lambda l: "".join([itos[i] for i in l])
    return stoi, itos, encode_fn, decode_fn, vocab_size, hp


def load_model(model_path: str, vocab_size: int, hp: dict):
    """加载模型（兼容 checkpoint dict 和裸 state_dict）"""
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        iter_num = ckpt.get("iter", "?")
        print(f"检查点步数: {iter_num}")
    else:
        state_dict = ckpt

    model = GPT(
        vocab_size=vocab_size,
        n_embd=hp["n_embd"],
        n_layer=hp["n_layer"],
        n_head=hp["n_head"],
        block_size=hp["block_size"],
        dropout=hp["dropout"],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_text(encode_fn, decode_fn, vocab_stoi, model, prompt="", max_tokens=200,
                  temperature=0.8, top_k=40):
    """从 prompt 开始自回归生成"""
    if prompt:
        filtered = "".join(c for c in prompt if c in vocab_stoi)
        if not filtered:
            print("⚠ 提示文本中所有字符都不在词汇表中，使用空上下文")
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            context = torch.tensor([encode_fn(filtered)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_tokens,
                                   temperature=temperature, top_k=top_k)
    return decode_fn(generated[0].tolist())


def interactive_mode(model, encode_fn, decode_fn):
    """交互式生成"""
    print("=" * 60)
    print("  字符级 GPT 交互式生成")
    print("  'q' / 'quit' 退出")
    print("  ':temp 0.7' 调整温度，':topk 50' 调整 top-k")
    print("  ':tokens 300' 调整生成长度")
    print("=" * 60)

    temp = 0.8
    top_k = 40
    max_tokens = 200

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            break

        if user_input.startswith(":temp "):
            try:
                temp = float(user_input.split()[1])
                print(f"温度 → {temp}")
            except ValueError:
                print(f"无效: {user_input}")
            continue
        if user_input.startswith(":topk "):
            try:
                top_k = int(user_input.split()[1])
                print(f"top_k → {top_k}")
            except ValueError:
                print(f"无效: {user_input}")
            continue
        if user_input.startswith(":tokens "):
            try:
                max_tokens = int(user_input.split()[1])
                print(f"最大 token → {max_tokens}")
            except ValueError:
                print(f"无效: {user_input}")
            continue

        print(f"\n{'─' * 40}")
        result = generate_text(encode_fn, decode_fn, stoi, model, prompt=user_input,
                               max_tokens=max_tokens, temperature=temp, top_k=top_k)
        print(result)
        print(f"{'─' * 40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="字符级 GPT 推理")
    parser.add_argument("prompt", nargs="?", default=None,
                        help="提示文本（不提供则进入交互模式）")
    parser.add_argument("--tokens", type=int, default=200, help="最大生成长度")
    parser.add_argument("--temp", type=float, default=0.8, help="温度 (0=确定)")
    parser.add_argument("--topk", type=int, default=40, help="top-k 采样")
    parser.add_argument("--model", type=str, default=os.path.join(_SCRIPT_DIR, "mini_gpt.pt"),
                        help="模型文件路径 (默认: mini_gpt.pt)")
    parser.add_argument("--vocab", type=str, default=os.path.join(_SCRIPT_DIR, "vocab.json"),
                        help="词汇表文件路径 (默认: vocab.json)")
    args = parser.parse_args()

    print(f"设备: {device}")

    stoi, itos, encode_fn, decode_fn, vocab_size, hp = load_vocab_and_hp(args.vocab)
    model = load_model(args.model, vocab_size, hp)
    print(f"超参数: n_embd={hp['n_embd']}, n_layer={hp['n_layer']}, "
          f"n_head={hp['n_head']}, block_size={hp['block_size']}")
    print(f"词汇表大小: {vocab_size}")
    print("模型加载成功！\n")

    if args.prompt:
        print(generate_text(encode_fn, decode_fn, stoi, model, prompt=args.prompt,
                            max_tokens=args.tokens, temperature=args.temp,
                            top_k=args.topk))
    else:
        interactive_mode(model, encode_fn, decode_fn)