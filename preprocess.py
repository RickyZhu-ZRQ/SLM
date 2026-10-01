"""
语料预处理：parquet 对话/文本文件 → 纯文本 txt 文件
"""

import os
import glob
import concurrent.futures
from tqdm import tqdm
import pyarrow.parquet as pq

# ========== 配置 ==========
CORPUS_FOLDER = r"C:\Users\Think\Desktop\input"   # 原始 Parquet 文件夹路径
OUTPUT_FOLDER = "input"                            # 输出 txt 文件夹（脚本所在目录下）
NUM_WORKERS = 4                                    # 并行进程数，建议不超过 CPU 核心数
# ==========================

# 错误计数器（进程间共享需用 multiprocessing.Value，这里走简化方案：返回状态）
SKIPPED_FILES = []
ERROR_FILES = []


def process_parquet(file_path: str, output_dir: str) -> dict:
    """
    处理单个 Parquet 文件，提取文本并保存为同名 .txt 文件。
    返回 {"ok": True/False, "path": ..., "error": ...}
    """
    text = ""
    try:
        table = pq.read_table(file_path)
        columns = table.column_names

        if "conversations" in columns:
            df = table.select(["conversations"]).to_pandas()
            for conversations in df["conversations"]:
                if isinstance(conversations, list):
                    for turn in conversations:
                        if isinstance(turn, dict) and "content" in turn:
                            role = turn.get("role", "unknown")
                            content = turn.get("content", "")
                            text += f"<|{role}|>: {content}\n"
                        elif isinstance(turn, str):
                            text += turn + "\n"
                else:
                    text += str(conversations) + "\n"
        elif "text" in columns:
            df = table.select(["text"]).to_pandas()
            text += "\n".join(df["text"].astype(str).tolist()) + "\n"
        elif "source" in columns and "answer" in columns:
            df = table.select(["source", "answer"]).to_pandas()
            for _, row in df.iterrows():
                q = str(row.get("source", ""))
                a = str(row.get("answer", ""))
                text += f"问题: {q}\n答案: {a}\n"

    except Exception as e:
        return {"ok": False, "path": file_path, "error": str(e)}

    if text:
        base_name = os.path.basename(file_path)
        output_name = os.path.splitext(base_name)[0] + ".txt"
        output_path = os.path.join(output_dir, output_name)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return {"ok": True, "path": file_path, "chars": len(text)}
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

    print(f"找到 {len(file_list)} 个 Parquet 文件，使用 {NUM_WORKERS} 个进程并行处理...")
    print()

    success_count = 0
    fail_count = 0
    total_chars = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_parquet, path, OUTPUT_FOLDER): path
            for path in file_list
        }
        with tqdm(total=len(file_list), desc="处理文件", unit="file") as pbar:
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