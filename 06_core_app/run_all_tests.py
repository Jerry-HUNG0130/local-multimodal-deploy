#!/usr/bin/env python3
"""
完整推論腳本：從 Llama3 原始模型 (mxbai-embed-large) 到各階段改動
一鍵執行所有 10 輪本地推論，每輪產出 answers.json + hardware.json
評分由 Claude Code 後續讀取 JSON 進行，不呼叫任何外部 API
"""
import os
import sys
import json
import glob
import shutil
import subprocess
import time
from datetime import datetime

# 確保可以 import evaluate_pipeline
sys.path.insert(0, os.path.dirname(__file__))

from evaluate_pipeline import (
    generate_answers, HardwareMonitor,
    RESULTS_DIR, HARDWARE_DIR
)

# ============================================================
# 路徑設定
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "04_knowledge_engine")
DOCS_DIR = os.path.join(BASE_DIR, "01_docs_library")
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_dataset.json")

# 向量庫路徑
DB_V1_MXBAI = os.path.join(KNOWLEDGE_DIR, "vector_db_v1_mxbai")
DB_V1_BGE = os.path.join(KNOWLEDGE_DIR, "vector_db")
DB_V2_BGE = os.path.join(KNOWLEDGE_DIR, "vector_db_v2")


# ============================================================
# 測試配置（10 輪）
# ============================================================
TEST_CONFIGS = [
    # --- 原始基線 ---
    {
        "run_label": "original_v1-mxbai_k3_rw-keywords",
        "embed": "mxbai",
        "db_dir": DB_V1_MXBAI,
        "db_version": "v1",
        "k": 3,
        "rewrite": "keywords",
        "desc": "原始基線：v1 + mxbai-embed-large + k=3 + keywords",
        "round": "Original",
    },
    # --- Embedding 切換 ---
    {
        "run_label": "baseline_v1-bge_k3_rw-keywords",
        "embed": "bge-m3",
        "db_dir": DB_V1_BGE,
        "db_version": "v1",
        "k": 3,
        "rewrite": "keywords",
        "desc": "Embedding 切換：v1 + bge-m3 + k=3 + keywords",
        "round": "Baseline",
    },
    # --- Round 1: k-value tuning ---
    {
        "run_label": "R1_dbv1_k5_rw-keywords",
        "embed": "bge-m3",
        "db_dir": DB_V1_BGE,
        "db_version": "v1",
        "k": 5,
        "rewrite": "keywords",
        "desc": "R1：v1 + bge-m3 + k=5 + keywords",
        "round": "R1",
    },
    {
        "run_label": "R1_dbv1_k7_rw-keywords",
        "embed": "bge-m3",
        "db_dir": DB_V1_BGE,
        "db_version": "v1",
        "k": 7,
        "rewrite": "keywords",
        "desc": "R1：v1 + bge-m3 + k=7 + keywords",
        "round": "R1",
    },
    {
        "run_label": "R1_dbv1_k10_rw-keywords",
        "embed": "bge-m3",
        "db_dir": DB_V1_BGE,
        "db_version": "v1",
        "k": 10,
        "rewrite": "keywords",
        "desc": "R1：v1 + bge-m3 + k=10 + keywords",
        "round": "R1",
    },
    # --- Round 2: Chunking 策略 (v2 MarkdownHeader) ---
    {
        "run_label": "R2_dbv2_k3_rw-keywords",
        "embed": "bge-m3",
        "db_dir": DB_V2_BGE,
        "db_version": "v2",
        "k": 3,
        "rewrite": "keywords",
        "desc": "R2：v2 MarkdownHeader + bge-m3 + k=3 + keywords",
        "round": "R2",
    },
    {
        "run_label": "R2_dbv2_k5_rw-keywords",
        "embed": "bge-m3",
        "db_dir": DB_V2_BGE,
        "db_version": "v2",
        "k": 5,
        "rewrite": "keywords",
        "desc": "R2：v2 + bge-m3 + k=5 + keywords",
        "round": "R2",
    },
    {
        "run_label": "R2_dbv2_k7_rw-keywords",
        "embed": "bge-m3",
        "db_dir": DB_V2_BGE,
        "db_version": "v2",
        "k": 7,
        "rewrite": "keywords",
        "desc": "R2：v2 + bge-m3 + k=7 + keywords",
        "round": "R2",
    },
    # --- Round 3: Query Rewrite 策略 ---
    {
        "run_label": "R3_dbv2_k3_rw-sentence",
        "embed": "bge-m3",
        "db_dir": DB_V2_BGE,
        "db_version": "v2",
        "k": 3,
        "rewrite": "sentence",
        "desc": "R3：v2 + bge-m3 + k=3 + sentence rewrite",
        "round": "R3",
    },
    {
        "run_label": "R3_dbv2_k3_rw-dual",
        "embed": "bge-m3",
        "db_dir": DB_V2_BGE,
        "db_version": "v2",
        "k": 3,
        "rewrite": "dual",
        "desc": "R3：v2 + bge-m3 + k=3 + dual rewrite",
        "round": "R3",
    },
]


# ============================================================
# 向量庫建構
# ============================================================
def _create_embeddings(embed_model):
    """建立 embedding 模型實例"""
    if embed_model == "mxbai":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="mxbai-embed-large")
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cuda"},
        )


def build_vector_db(version, embed_model, db_dir):
    """建構向量庫"""
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    print(f"\n{'='*60}")
    print(f"🔨 建構向量庫：version={version}, embed={embed_model}")
    print(f"   輸出路徑：{db_dir}")
    print(f"{'='*60}")

    # 載入並切分文件
    if version == "v2":
        sys.path.insert(0, KNOWLEDGE_DIR)
        from embed_docs import load_and_split_markdown
        chunks = load_and_split_markdown(DOCS_DIR)
    else:
        from langchain_community.document_loaders import TextLoader
        docs = []
        for fp in sorted(glob.glob(os.path.join(DOCS_DIR, "*.md"))):
            loader = TextLoader(fp, encoding="utf-8")
            docs.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

    print(f"  ✂️  共 {len(chunks)} 個 chunks")
    embeddings = _create_embeddings(embed_model)
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_dir)
    print(f"  ✅ 向量庫建構完成：{db_dir}")


# ============================================================
# 自訂 init_rag（支援不同 embedding 模型 + 自訂 DB 路徑）
# ============================================================
def init_rag_custom(k, db_dir, embed_model):
    """初始化 RAG 引擎，支援 mxbai 和 bge-m3"""
    from langchain_community.vectorstores import Chroma
    from langchain_community.llms import Ollama

    print(f"📦 載入 RAG 引擎 (k={k}, embed={embed_model}, db={os.path.basename(db_dir)})...")
    embeddings = _create_embeddings(embed_model)
    vector_db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    llm = Ollama(model="llama3", temperature=0.0)
    print(f"✅ RAG 引擎就緒\n")
    return retriever, llm


# ============================================================
# 單輪推論執行（僅推論 + 硬體監控，不做評分）
# ============================================================
def run_single_inference(test_cases, config):
    """執行單輪推論，產出 answers.json + hardware.json"""
    run_label = config["run_label"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'#'*60}")
    print(f"  🚀 {config['desc']}")
    print(f"  run_label: {run_label}")
    print(f"{'#'*60}\n")

    # 啟動硬體監控
    hw_monitor = HardwareMonitor(interval=5)
    hw_monitor.start()
    run_start = time.time()

    # 初始化 RAG
    retriever, llm = init_rag_custom(
        k=config["k"],
        db_dir=config["db_dir"],
        embed_model=config["embed"],
    )

    # 產生回答
    inference_start = time.time()
    answer_results = generate_answers(
        test_cases, retriever, llm,
        rewrite_mode=config["rewrite"],
        k=config["k"],
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
            "embed_model": config["embed"],
            "db_version": config["db_version"],
            "k": config["k"],
            "rewrite_mode": config["rewrite"],
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

    return {
        "run_label": run_label,
        "round": config["round"],
        "desc": config["desc"],
        "timestamp": timestamp,
        "timing": hw_report["timing"],
        "hardware": hw_summary,
        "files": {
            "answers": os.path.basename(json_path),
            "hardware": os.path.basename(hw_path),
        },
    }


# ============================================================
# 主流程
# ============================================================
def main():
    total_start = time.time()

    print("=" * 60)
    print("  🏁 完整重新推論：10 輪全跑（僅推論，不評分）")
    print("  評分將由 Claude Code 後續讀取 JSON 進行")
    print("=" * 60)

    # 載入測試資料
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"📋 載入 {len(test_cases)} 題測試資料\n")

    # ---- Phase 1: 確認 Ollama 模型 ----
    print("=" * 60)
    print("  Phase 1: 確認模型")
    print("=" * 60)
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if "mxbai-embed-large" not in result.stdout:
        print("📥 正在拉取 mxbai-embed-large...")
        subprocess.run(["ollama", "pull", "mxbai-embed-large"], check=True)
    print("✅ mxbai-embed-large 就緒")
    if "llama3" not in result.stdout:
        print("❌ llama3 未安裝！請先執行 ollama pull llama3")
        return
    print("✅ llama3 就緒")

    # ---- Phase 2: 建構向量庫 ----
    print("\n" + "=" * 60)
    print("  Phase 2: 建構向量庫（3 個）")
    print("=" * 60)

    build_vector_db(version="v1", embed_model="mxbai", db_dir=DB_V1_MXBAI)
    build_vector_db(version="v1", embed_model="bge-m3", db_dir=DB_V1_BGE)
    build_vector_db(version="v2", embed_model="bge-m3", db_dir=DB_V2_BGE)

    # ---- Phase 3: 清空舊測試結果 ----
    print("\n" + "=" * 60)
    print("  Phase 3: 清空舊測試結果")
    print("=" * 60)

    for d in [RESULTS_DIR, HARDWARE_DIR]:
        if os.path.exists(d):
            old_files = [f for f in os.listdir(d) if not f.startswith(".")]
            for f_name in old_files:
                os.remove(os.path.join(d, f_name))
            print(f"  🗑️  已清除 {d} ({len(old_files)} 個檔案)")
        os.makedirs(d, exist_ok=True)

    # ---- Phase 4: 執行 10 輪推論 ----
    print("\n" + "=" * 60)
    print(f"  Phase 4: 開始推論（共 {len(TEST_CONFIGS)} 輪 × {len(test_cases)} 題）")
    print("=" * 60)

    all_results = []
    for i, config in enumerate(TEST_CONFIGS):
        print(f"\n{'*'*60}")
        print(f"  [{i+1}/{len(TEST_CONFIGS)}] {config['desc']}")
        print(f"{'*'*60}")

        result = run_single_inference(test_cases, config)
        all_results.append(result)

    # ---- Phase 5: 總結 ----
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("  Phase 5: 全部推論完成！")
    print("=" * 60)

    header = f"{'Round':<12}{'Config':<40}{'推論(s)':<10}{'每題(s)':<10}{'GPU peak(MB)':<14}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        if r is None:
            continue
        row = (f"{r['round']:<12}"
               f"{r['run_label'][:39]:<40}"
               f"{r['timing']['inference_seconds']:<10}"
               f"{r['timing']['avg_inference_per_question']:<10}"
               f"{r['hardware'].get('gpu_mem_used_max_mb', '-'):<14}")
        print(row)

    print(f"\n⏱️  全部推論總耗時：{total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"\n📂 產出檔案位於：")
    print(f"   answers.json → {RESULTS_DIR}")
    print(f"   hardware.json → {HARDWARE_DIR}")
    print(f"\n✅ 推論完成！請由 Claude Code 讀取 answers.json 進行評分。")


if __name__ == "__main__":
    main()
