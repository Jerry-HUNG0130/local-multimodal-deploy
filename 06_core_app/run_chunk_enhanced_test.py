#!/usr/bin/env python3
"""
Round 8: Chunk 內容增強測試
分別測試 v3 (MarkdownHeaderTextSplitter + 標題前綴) 和 v4 (MarkdownNodeParser + 標題前綴)
使用 R7 最佳配置：sentence rewrite + HyDE + BM25 hybrid + reranker
"""
import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from evaluate_pipeline import (
    generate_answers, HardwareMonitor, build_bm25_index, init_rag,
    RESULTS_DIR, HARDWARE_DIR
)

DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_dataset.json")


def run_test(db_version, run_label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(f"  🚀 Round 8: Chunk 增強測試 — {db_version}")
    print(f"  配置: {db_version} + sentence rewrite + HyDE + BM25 + reranker")
    print("=" * 60)

    # 載入測試資料
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"📋 載入 {len(test_cases)} 題測試資料\n")

    # 初始化 RAG（使用 evaluate_pipeline 的 init_rag）
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.llms import Ollama

    print(f"📦 載入 RAG 引擎 (db={db_version})...")
    retriever, llm = init_rag(k=3, db_version=db_version)

    # 取得 vector_db 物件以建立 BM25 索引
    db_map = {
        'v3': os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_knowledge_engine", "vector_db_v3"),
        'v4': os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_knowledge_engine", "vector_db_v4"),
    }
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cuda"})
    vector_db = Chroma(persist_directory=db_map[db_version], embedding_function=embeddings)

    # 建立 BM25 索引
    print("📚 建立 BM25 索引...")
    bm25_index, bm25_docs = build_bm25_index(vector_db)

    # 初始化 Reranker
    print("🔄 載入 Reranker (BAAI/bge-reranker-v2-m3)...")
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda")
    print("✅ Reranker 就緒\n")

    # 啟動硬體監控
    hw_monitor = HardwareMonitor(interval=5)
    hw_monitor.start()
    run_start = time.time()

    # 產生回答（啟用 HyDE）
    inference_start = time.time()
    answer_results = generate_answers(
        test_cases, retriever, llm,
        rewrite_mode="sentence",
        k=3,
        reranker=reranker,
        bm25_index=bm25_index,
        bm25_docs=bm25_docs,
        use_hyde=True,
    )
    inference_duration = time.time() - inference_start
    total_duration = time.time() - run_start
    hw_monitor.stop()

    # 儲存檔案
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HARDWARE_DIR, exist_ok=True)

    # answers.json
    answers_json = [
        {
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "answer": r["answer"],
            "context": r["context"],
        }
        for r in answer_results
    ]
    json_path = os.path.join(RESULTS_DIR, f"{run_label}_{timestamp}_answers.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(answers_json, f, ensure_ascii=False, indent=2)
    print(f"💾 模型回答：{json_path}")

    # hardware.json
    hw_summary = hw_monitor.summary()
    hw_report = {
        "run_label": run_label,
        "timestamp": timestamp,
        "config": {
            "embed_model": "bge-m3",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "bm25": True,
            "hyde": True,
            "db_version": db_version,
            "chunk_enhanced": True,
            "k": 3,
            "fetch_k": 9,
            "rewrite_mode": "sentence",
            "num_questions": len(test_cases),
        },
        "timing": {
            "total_seconds": round(total_duration, 1),
            "inference_seconds": round(inference_duration, 1),
            "avg_inference_per_question": round(inference_duration / len(test_cases), 2),
        },
        "hardware": hw_summary,
        "samples": hw_monitor.samples,
    }
    hw_path = os.path.join(HARDWARE_DIR, f"{run_label}_{timestamp}_hardware.json")
    with open(hw_path, "w", encoding="utf-8") as f:
        json.dump(hw_report, f, ensure_ascii=False, indent=2)
    print(f"💾 硬體監控：{hw_path}")

    # 摘要
    print(f"\n⚙️  【推論完成 — {db_version}】")
    print(f"  總耗時：{total_duration:.0f}s")
    print(f"  平均每題：{inference_duration / len(test_cases):.1f}s")
    if "gpu_util_avg" in hw_summary:
        print(f"  GPU 使用率：avg {hw_summary['gpu_util_avg']}% / max {hw_summary['gpu_util_max']}%")
    if "gpu_mem_used_max_mb" in hw_summary:
        print(f"  GPU 記憶體：avg {hw_summary['gpu_mem_used_avg_mb']}MB / max {hw_summary['gpu_mem_used_max_mb']}MB")

    return json_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True, choices=['v3', 'v4'],
                        help='v3=HeaderSplitter+標題前綴, v4=NodeParser+標題前綴')
    args = parser.parse_args()

    if args.db == 'v3':
        run_label = "R8a_dbv3_k3_rw-sentence_hyde_hybrid-bm25_reranker"
    else:
        run_label = "R8b_dbv4_k3_rw-sentence_hyde_hybrid-bm25_reranker"

    run_test(args.db, run_label)
    print(f"\n✅ 推論完成！請由 Claude Code 讀取 answers.json 進行評分。")


if __name__ == "__main__":
    main()
