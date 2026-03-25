#!/usr/bin/env python3
"""
評測輔助腳本：
1. read_batch: 從 answers.json 讀取指定批次的題目（精簡輸出供 Claude Code 評分）
2. write_scores: 將評分結果寫入 CSV
3. compile_report: 彙整所有 CSV 並產出 PNG 圖表
"""
import os
import sys
import json
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

SCORES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "07_evaluation_results", "scores")


def read_batch(answers_file, start, count):
    """讀取 answers.json 的指定批次，精簡輸出"""
    with open(answers_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    end = min(start + count, len(data))
    batch = data[start:end]

    for i, entry in enumerate(batch):
        idx = start + i + 1
        # 截斷 context 以節省 token
        ctx = entry["context"][:200] + "..." if len(entry["context"]) > 200 else entry["context"]
        print(f"--- Q{idx} ---")
        print(f"問: {entry['question']}")
        print(f"標準答案: {entry['ground_truth']}")
        print(f"模型回答: {entry['answer'][:300]}")
        print(f"檢索內容(前200字): {ctx}")
        print()

    print(f"[共 {end - start} 題，index {start+1}~{end}]")


def write_scores(output_csv, scores_json_str):
    """將 JSON 格式的評分寫入 CSV"""
    scores = json.loads(scores_json_str)
    df = pd.DataFrame(scores)

    # 如果 CSV 已存在，追加
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 已寫入 {len(scores)} 筆評分至 {output_csv}（目前共 {len(df)} 筆）")


def compile_report(csv_path, run_label):
    """從 CSV 產出 PNG 圖表"""
    from evaluate_pipeline import generate_visual_report

    df = pd.read_csv(csv_path)
    png_path = csv_path.replace(".csv", ".png")
    generate_visual_report(df, output_path=png_path, run_label=run_label)

    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    print(f"\n📊 【{run_label} 評測結果】")
    for m in metrics:
        print(f"  {m}: {df[m].mean():.4f}")
    print(f"✅ 圖表已產出：{png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("read")
    p1.add_argument("file", help="answers.json 路徑")
    p1.add_argument("--start", type=int, default=0)
    p1.add_argument("--count", type=int, default=25)

    p2 = sub.add_parser("write")
    p2.add_argument("csv", help="輸出 CSV 路徑")
    p2.add_argument("scores", help="JSON 格式的評分字串")

    p3 = sub.add_parser("report")
    p3.add_argument("csv", help="CSV 路徑")
    p3.add_argument("label", help="run_label")

    args = parser.parse_args()
    if args.cmd == "read":
        read_batch(args.file, args.start, args.count)
    elif args.cmd == "write":
        write_scores(args.csv, args.scores)
    elif args.cmd == "report":
        compile_report(args.csv, args.label)
