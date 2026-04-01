#!/usr/bin/env python3
"""
Round 13: Contextual Retrieval (v5)
在 R12d 最佳配置基礎上，將 vector DB 從 v3（標題前綴）升級為 v5（標題前綴 + LLM 語義定位）。
Embedding 前由 Qwen2.5 為每個 chunk 生成 1-2 句語義描述，拼接到 chunk 前面再做 embedding。
其餘配置不變：sentence rewrite + HyDE + BM25 hybrid + reranker + few-shot v3 prompt。
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


def run_test():
    run_label = "R13_dbv5_k3_rw-sentence_hyde_hybrid-bm25_reranker_fewshot_qwen25_contextual"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_version = "v5"
    llm_model = "qwen2.5:7b-instruct-q8_0"

    print("=" * 60)
    print(f"  Round 13: Contextual Retrieval (v5)")
    print(f"  配置: {db_version} + sentence + HyDE + BM25 + reranker + few-shot v3")
    print(f"  LLM: {llm_model}")
    print(f"  改動: v3 標題前綴 → v5 標題前綴 + LLM 語義定位描述")
    print("=" * 60)

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"載入 {len(test_cases)} 題測試資料\n")

    retriever, llm = init_rag(k=3, db_version=db_version, llm_model=llm_model)

    # 建立 BM25 索引（從 v5 的 ChromaDB 讀取）
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_knowledge_engine", "vector_db_v5")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cuda"})
    vector_db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

    print("建立 BM25 索引...")
    bm25_index, bm25_docs = build_bm25_index(vector_db)

    print("載入 Reranker...")
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda")
    print("就緒\n")

    hw_monitor = HardwareMonitor(interval=5)
    hw_monitor.start()
    run_start = time.time()

    inference_start = time.time()
    answer_results = generate_answers(
        test_cases, retriever, llm,
        rewrite_mode="sentence",
        k=3,
        reranker=reranker,
        bm25_index=bm25_index,
        bm25_docs=bm25_docs,
        use_hyde=True,
        prompt_version='v3',
    )
    inference_duration = time.time() - inference_start
    total_duration = time.time() - run_start
    hw_monitor.stop()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HARDWARE_DIR, exist_ok=True)

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
    print(f"模型回答：{json_path}")

    hw_summary = hw_monitor.summary()
    hw_report = {
        "run_label": run_label,
        "timestamp": timestamp,
        "config": {
            "llm_model": llm_model,
            "embed_model": "bge-m3",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "bm25": True,
            "hyde": True,
            "db_version": db_version,
            "k": 3,
            "fetch_k": 9,
            "rewrite_mode": "sentence",
            "prompt_version": "v3-fewshot",
            "contextual_retrieval": True,
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
    print(f"硬體監控：{hw_path}")

    print(f"\n【推論完成 — Contextual Retrieval (v5)】")
    print(f"  總耗時：{total_duration:.0f}s")
    print(f"  平均每題：{inference_duration / len(test_cases):.1f}s")
    if "gpu_util_avg" in hw_summary:
        print(f"  GPU 使用率：avg {hw_summary['gpu_util_avg']}% / max {hw_summary['gpu_util_max']}%")
    if "gpu_mem_used_max_mb" in hw_summary:
        print(f"  GPU 記憶體：avg {hw_summary['gpu_mem_used_avg_mb']}MB / max {hw_summary['gpu_mem_used_max_mb']}MB")

    return json_path


if __name__ == "__main__":
    result_path = run_test()
    print(f"\n推論完成！請由 Claude Code 讀取 answers.json 進行評分。")
    print(f"   檔案路徑：{result_path}")
