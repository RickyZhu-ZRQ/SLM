"""
语料预处理：parquet 对话/文本文件 → 纯文本 txt 文件
"""

import gc
import os
import re
import sys
import glob
import unicodedata
import concurrent.futures
from tqdm import tqdm
import pyarrow.parquet as pq

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 配置 ==========
CORPUS_FOLDER = r"C:\Users\Think\Desktop\input"   # 原始 Parquet 文件夹路径
OUTPUT_FOLDER = os.path.join(_SCRIPT_DIR, "input") # 输出 txt 文件夹（脚本所在目录下）
NUM_WORKERS = 4                                    # 并行进程数，建议不超过 CPU 核心数
# ==========================

# ========== 文本清洗 ==========
_RE_CONTROL = re.compile(
    "[" "\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f" "]", flags=re.UNICODE
)

def clean_text(text: str) -> str:
    """移除不可渲染/干扰字符，保留正常 Unicode 文本"""
    text = _RE_CONTROL.sub("", text)
    lines = []
    for line in text.split("\n"):
        cleaned = []
        for ch in line:
            cp = ord(ch)
            # 私有区 (PUA)
            if (0xE000 <= cp <= 0xF8FF) or (0xF0000 <= cp <= 0x10FFFD):
                continue
            # 非字符 (noncharacters)
            if (cp & 0xFFFE) == 0xFFFE and (0xFDD0 <= cp <= 0xFDEF or cp >= 0xFFFE):
                continue
            # 格式字符（BOM、零宽等）
            if unicodedata.category(ch) == "Cf" or unicodedata.category(ch) == "Cn":
                continue
            cleaned.append(ch)
        lines.append("".join(cleaned))
    return "\n".join(lines)
# ============================

# 错误计数器（进程间共享需用 multiprocessing.Value，这里走简化方案：返回状态）
SKIPPED_FILES = []
ERROR_FILES = []


# 分块写入阈值：pieces 超过此数量就先 join 写入并清空
CHUNK_PIECES = 100_000
# 分块写入缓冲区：join 后超过此字节数也触发写入
CHUNK_BYTES = 4 * 1024 * 1024  # 4MB


def _flush_pieces(pieces: list, f_out) -> None:
    """将 pieces 列表 join、清洗、写入文件并清空"""
    if not pieces:
        return
    text = "".join(pieces)
    text = clean_text(text)
    f_out.write(text)
    pieces.clear()


def process_parquet(file_path: str, output_dir: str) -> dict:
    """
    处理单个 Parquet 文件，提取文本并保存为同名 .txt 文件。
    返回 {"ok": True/False, "path": ..., "error": ...}
    """
    try:
        table = pq.read_table(file_path, memory_map=True)
        columns = table.column_names

        base_name = os.path.basename(file_path)
        output_name = os.path.splitext(base_name)[0] + ".txt"
        output_path = os.path.join(output_dir, output_name)

        total_chars = 0
        pieces = []  # 流式缓冲区

        with open(output_path, "w", encoding="utf-8") as f_out:

            if "conversations" in columns:
                conv_list = table.column("conversations").to_pylist()
                for conversations in conv_list:
                    if isinstance(conversations, list):
                        for turn in conversations:
                            if isinstance(turn, dict) and "content" in turn:
                                role = turn.get("role", "unknown")
                                pieces.append(f"<|{role}|>: {turn['content']}\n")
                            elif isinstance(turn, str):
                                pieces.append(turn + "\n")
                    else:
                        pieces.append(str(conversations) + "\n")
                    if len(pieces) >= CHUNK_PIECES:
                        _flush_pieces(pieces, f_out)

            elif "text" in columns:
                texts = table.column("text").to_pylist()
                # 分批写入，避免列表过大
                for i in range(0, len(texts), CHUNK_PIECES):
                    chunk = texts[i:i + CHUNK_PIECES]
                    chunk_text = "".join(
                        (str(t) + "\n") for t in chunk if t is not None
                    )
                    f_out.write(chunk_text)
                    total_chars += len(chunk_text)

            elif "source" in columns and "answer" in columns:
                sources = table.column("source").to_pylist()
                answers = table.column("answer").to_pylist()
                for q, a in zip(sources, answers):
                    if q is not None or a is not None:
                        pieces.append(f"问题: {q}\n答案: {a}\n")
                    if len(pieces) >= CHUNK_PIECES:
                        _flush_pieces(pieces, f_out)

            # 最后一批
            _flush_pieces(pieces, f_out)

        # 统计实际写入量
        if "text" not in columns:
            try:
                total_chars = os.path.getsize(output_path)
            except OSError:
                pass

        if total_chars == 0:
            # 空文件：检查是否真的有内容
            try:
                total_chars = os.path.getsize(output_path)
            except OSError:
                total_chars = 0

    except Exception as e:
        return {"ok": False, "path": file_path, "error": str(e)}
    finally:
        del table
        gc.collect()

    if total_chars > 0:
        return {"ok": True, "path": file_path, "chars": total_chars}
    else:
        return {"ok": False, "path": file_path, "error": "提取到空文本"}


def main():
    # ======== 运行时 Python 版本信息 ========
    import sys
    print(f"Python {sys.version}")
    print(f"当前脚本路径: {os.path.abspath(__file__)}")
    print()

    if not os.path.isdir(CORPUS_FOLDER):
        print(f"原始语料文件夹不存在: {CORPUS_FOLDER}")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(CORPUS_FOLDER, "**", "*.parquet"), recursive=True))
    if not file_list:
        # 也尝试 .parquet 不区分大小写
        file_list = sorted(glob.glob(os.path.join(CORPUS_FOLDER, "**", "*.[pP][aA][rR][qQ][uU][eE][tT]"), recursive=True))
    if not file_list:
        # 也检查根目录
        file_list = sorted(glob.glob(os.path.join(CORPUS_FOLDER, "*.parquet")))
    if not file_list:
        print("没有找到 Parquet 文件")
        return

    # 中断恢复：跳过已存在的 .txt（支持 Ctrl+C 后重跑）
    pending = []
    skipped_count = 0
    for path in file_list:
        base_name = os.path.basename(path)
        output_name = os.path.splitext(base_name)[0] + ".txt"
        if os.path.isfile(os.path.join(OUTPUT_FOLDER, output_name)):
            skipped_count += 1
        else:
            pending.append(path)

    if skipped_count > 0:
        print(f"跳过 {skipped_count} 个已存在的 txt 文件")

    if not pending:
        print("所有文件均已处理完成。")
        return

    print(f"待处理 {len(pending)} 个 Parquet 文件（共 {len(file_list)}），"
          f"使用 {NUM_WORKERS} 个进程并行处理...")
    print()

    success_count = 0
    fail_count = 0
    total_chars = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_parquet, path, OUTPUT_FOLDER): path
            for path in pending
        }
        with tqdm(total=len(pending), desc="处理文件", unit="file") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    fp = futures[future]
                    result = {"ok": False, "path": fp, "error": str(e)}

                if result["ok"]:
                    success_count += 1
                    total_chars += result.get("chars", 0)
                else:
                    fail_count += 1
                    ERROR_FILES.append(result)

                pbar.set_postfix_str(
                    f"成功 {success_count} | 失败 {fail_count} | 总字符 {total_chars:,}"
                )
                pbar.update(1)

    print()
    print(f"处理完成！成功: {success_count}, 失败: {fail_count}")
    if fail_count > 0:
        print("失败文件列表：")
        for err in ERROR_FILES:
            print(f"  - {err['path']}: {err['error']}")
    print(f"输出目录: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"总输出字符数: {total_chars:,}")


if __name__ == "__main__":
    main()