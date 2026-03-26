#!/usr/bin/env python3
"""
Round 7: HyDE (Hypothetical Document Embedding) 測試
在 Round 6 最佳配置基礎上加入 HyDE

檢索策略：
  1. sentence rewrite → 正式書面語查詢（供 BM25）
  2. HyDE → LLM 產生假設性規章回答（供 Vector search）
  3. Vector search(假答案, top 9) + BM25 search(書面語, top 9)
     → RRF 合併 → bge-reranker-v2-m3 重排 → top 3 → LLM 正式回答
"""
import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from evaluate_pipeline import (
    generate_answers, HardwareMonitor, build_bm25_index,
    RESULTS_DIR, HARDWARE_DIR
)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "04_knowledge_engine")
DB_V2_BGE = os.path.join(KNOWLEDGE_DIR, "vector_db_v2")
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_dataset.json")


def main():
    run_label = "R7_dbv2_k3_rw-sentence_hyde_hybrid-bm25_reranker"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  🚀 Round 7: HyDE + Hybrid Search 測試")
    print("  配置: v2 + bge-m3 + sentence rewrite + HyDE + BM25 + reranker")
    print("  策略: sentence rewrite + HyDE → Vector(假答案) + BM25(書面語)")
    print("         → RRF → reranker(3) → LLM")
    print("=" * 60)

    # 載入測試資料
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"📋 載入 {len(test_cases)} 題測試資料\n")

    # 初始化 RAG
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.llms import Ollama

    print("📦 載入 RAG 引擎...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
    )
    vector_db = Chroma(persist_directory=DB_V2_BGE, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="llama3", temperature=0.0)
    print("✅ RAG 引擎就緒")

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
            "bm25_tokenizer": "jieba",
            "hyde": True,
            "rrf_k": 60,
            "db_version": "v2",
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
    print(f"\n⚙️  【推論完成】")
    print(f"  總耗時：{total_duration:.0f}s")
    print(f"  平均每題：{inference_duration / len(test_cases):.1f}s")
    if "gpu_util_avg" in hw_summary:
        print(f"  GPU 使用率：avg {hw_summary['gpu_util_avg']}% / max {hw_summary['gpu_util_max']}%")
    if "gpu_mem_used_max_mb" in hw_summary:
        print(f"  GPU 記憶體：avg {hw_summary['gpu_mem_used_avg_mb']}MB / max {hw_summary['gpu_mem_used_max_mb']}MB")

    print(f"\n✅ 推論完成！請由 Claude Code 讀取 answers.json 進行評分。")


if __name__ == "__main__":
    main()
