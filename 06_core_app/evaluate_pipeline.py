import os
import json
import glob
import argparse
import subprocess
import time
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import anthropic
from pydantic import BaseModel, Field

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# BM25 Hybrid Search
import jieba
from rank_bm25 import BM25Okapi

# -- Matplotlib Chinese Font --
font_path = 'NotoSansTC.otf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Config
# ============================================================
DB_DIR_V1 = os.path.join(os.path.dirname(__file__), "..", "04_knowledge_engine", "vector_db")
DB_DIR_V2 = os.path.join(os.path.dirname(__file__), "..", "04_knowledge_engine", "vector_db_v2")
DB_DIR_V3 = os.path.join(os.path.dirname(__file__), "..", "04_knowledge_engine", "vector_db_v3")
DB_DIR_V4 = os.path.join(os.path.dirname(__file__), "..", "04_knowledge_engine", "vector_db_v4")
DATASET_FILE = "golden_dataset.json"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "07_evaluation_results", "scores")
HARDWARE_DIR = os.path.join(os.path.dirname(__file__), "..", "07_evaluation_results", "hardware")

class EvaluationScore(BaseModel):
    context_precision: float = Field(description="Score between 0.0 and 1.0. Is the retrieved context relevant to the question?")
    context_recall: float = Field(description="Score between 0.0 and 1.0. Does the retrieved context contain the information needed to answer the question (compare with Ground Truth)?")
    faithfulness: float = Field(description="Score between 0.0 and 1.0. Is the model's answer factually derived from the retrieved context (no hallucination)?")
    answer_relevancy: float = Field(description="Score between 0.0 and 1.0. How well does the answer address the question directly?")
    reasoning: str = Field(description="Brief explanation of the scores given.")

# ============================================================
# Hardware Monitor
# ============================================================
def _sample_hardware():
    """取得當前硬體使用狀態（CPU%, RAM MB, GPU Util%, GPU Mem MB）"""
    snapshot = {"timestamp": datetime.now().isoformat()}
    try:
        import psutil
        snapshot["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        snapshot["ram_used_mb"] = round(mem.used / 1024 / 1024)
        snapshot["ram_total_mb"] = round(mem.total / 1024 / 1024)
    except ImportError:
        # fallback: read from /proc
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        meminfo = {}
        for line in lines:
            parts = line.split()
            meminfo[parts[0].rstrip(":")] = int(parts[1])
        total_mb = meminfo["MemTotal"] // 1024
        avail_mb = meminfo["MemAvailable"] // 1024
        snapshot["ram_used_mb"] = total_mb - avail_mb
        snapshot["ram_total_mb"] = total_mb
        snapshot["cpu_percent"] = None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            snapshot["gpu_util_percent"] = int(parts[0].strip())
            snapshot["gpu_mem_used_mb"] = int(parts[1].strip())
            snapshot["gpu_mem_total_mb"] = int(parts[2].strip())
    except Exception:
        pass

    return snapshot


class HardwareMonitor:
    """背景執行緒定期採樣硬體數據，在測試結束後彙整摘要"""

    def __init__(self, interval=5):
        self.interval = interval
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"📡 硬體監控已啟動（每 {self.interval} 秒採樣）")

    def _run(self):
        while not self._stop_event.is_set():
            self.samples.append(_sample_hardware())
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        # 最後再採一次
        self.samples.append(_sample_hardware())
        print(f"📡 硬體監控已停止，共 {len(self.samples)} 筆採樣")

    def summary(self):
        """彙整硬體使用摘要"""
        if not self.samples:
            return {}
        gpu_utils = [s["gpu_util_percent"] for s in self.samples if "gpu_util_percent" in s]
        gpu_mems = [s["gpu_mem_used_mb"] for s in self.samples if "gpu_mem_used_mb" in s]
        ram_used = [s["ram_used_mb"] for s in self.samples if "ram_used_mb" in s]
        cpu_pcts = [s["cpu_percent"] for s in self.samples if s.get("cpu_percent") is not None]

        summary = {
            "sample_count": len(self.samples),
            "ram_total_mb": self.samples[0].get("ram_total_mb"),
            "gpu_mem_total_mb": self.samples[0].get("gpu_mem_total_mb"),
        }
        if gpu_utils:
            summary["gpu_util_avg"] = round(np.mean(gpu_utils), 1)
            summary["gpu_util_max"] = max(gpu_utils)
        if gpu_mems:
            summary["gpu_mem_used_avg_mb"] = round(np.mean(gpu_mems))
            summary["gpu_mem_used_max_mb"] = max(gpu_mems)
        if ram_used:
            summary["ram_used_avg_mb"] = round(np.mean(ram_used))
            summary["ram_used_max_mb"] = max(ram_used)
        if cpu_pcts:
            summary["cpu_percent_avg"] = round(np.mean(cpu_pcts), 1)
            summary["cpu_percent_max"] = round(max(cpu_pcts), 1)
        return summary


# ============================================================
# Step 1: Generate Answers from Local RAG (Direct Call)
# ============================================================
# Embedding 模型對照表
EMBED_MODEL_MAP = {
    'bge-m3': 'BAAI/bge-m3',
    'bge-large-zh': 'BAAI/bge-large-zh-v1.5',
}

def _get_db_dir(db_version, embed_model='bge-m3'):
    """根據 db_version 和 embed_model 決定 vector DB 目錄"""
    db_map = {'v1': DB_DIR_V1, 'v2': DB_DIR_V2, 'v3': DB_DIR_V3, 'v4': DB_DIR_V4}
    base = db_map.get(db_version, DB_DIR_V2)
    if embed_model != 'bge-m3':
        base = base + f"_{embed_model.replace('-', '')}"
    return base

def init_rag(k=3, db_version='v2', embed_model='bge-m3'):
    db_dir = _get_db_dir(db_version, embed_model)
    model_name = EMBED_MODEL_MAP.get(embed_model, 'BAAI/bge-m3')
    print(f"📦 正在載入本地 RAG 引擎 (k={k}, db={db_version}, embed={embed_model})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'}
    )
    vector_db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    llm = Ollama(model="llama3", temperature=0.0)
    print(f"✅ RAG 引擎就緒 (k={k}, db={db_version}, embed={embed_model})\n")
    return retriever, llm

def _decompose_query(question):
    """Rule-based 拆分複合問題為多個子查詢（不需要額外 LLM 呼叫）"""
    import re

    # 不拆分的模式：A還是B 的對比型問題
    if re.search(r'還是|或是|或者', question):
        return [question]

    # 用中文問號拆分
    fragments = re.split(r'？', question)
    fragments = [f.strip() for f in fragments if f.strip() and len(f.strip()) >= 6]

    if len(fragments) <= 1:
        return [question]

    # 限制最多 3 個子查詢
    fragments = fragments[:3]

    # 判斷後續片段是否缺少主語，如果是則補上第一個片段的主題前綴
    # 提取第一個片段的主題（取到第一個逗號或問題核心之前的部分）
    first = fragments[0]
    # 嘗試提取主題前綴：「去外縣市拜訪客戶，每天吃飯...」→「去外縣市拜訪客戶」
    topic_match = re.match(r'^(.+?)[，,]', first)
    topic_prefix = topic_match.group(1) if topic_match else ""

    sub_queries = [first + "？"]
    for frag in fragments[1:]:
        # 判斷是否以功能詞開頭（缺少主語）
        if re.match(r'^(有|會|需|能|可|要|如果|沒|是否|多久|什麼|幾|怎)', frag):
            if topic_prefix:
                sub_queries.append(topic_prefix + "，" + frag + "？")
            else:
                sub_queries.append(frag + "？")
        else:
            sub_queries.append(frag + "？")

    return sub_queries


def _rewrite_keywords(llm, question):
    """原始策略：將問題轉換為 3~5 個關鍵字"""
    rewrite_prompt = (
        "請將以下員工口語問題，轉換為 3 到 5 個用來搜尋公司規章的『正式關鍵字』。\n"
        "【嚴格規定】：\n"
        "1. 絕對只能使用「繁體中文」輸出，禁止出現任何英文。\n"
        "2. 關鍵字之間請用空白鍵隔開即可，絕對不要使用項目符號(* 或 -)。\n"
        "3. 不要加入任何解釋、問候或開場白。\n"
        f"原始問題：'{question}'"
    )
    expanded_query = llm.invoke(rewrite_prompt).strip()
    if ":" in expanded_query:
        expanded_query = expanded_query.split(":")[-1].strip()
    for noise in ["關鍵字", "正式", "搜尋", "是："]:
        expanded_query = expanded_query.replace(noise, "")
    return expanded_query.strip()


def _rewrite_sentence(llm, question):
    """方案 B：將口語問題改寫為一句完整的正式書面語句子，保留語義完整性"""
    rewrite_prompt = (
        "你是公司規章檢索系統的查詢優化器。\n"
        "請將以下員工的口語問題，改寫為一句「完整的正式書面語句子」，用於搜尋公司內部規章。\n"
        "【嚴格規定】：\n"
        "1. 只輸出改寫後的一句話，不要加任何解釋或前綴。\n"
        "2. 必須使用繁體中文。\n"
        "3. 保留原始問題的完整語意，用正式的公司規章用語重新表達。\n"
        "4. 不要拆成關鍵字，必須是一句完整、通順的句子。\n"
        f"原始問題：'{question}'"
    )
    rewritten = llm.invoke(rewrite_prompt).strip()
    # 清洗：移除可能的引號包裹或前綴
    for prefix in ["改寫後：", "改寫：", "查詢：", "問題："]:
        if rewritten.startswith(prefix):
            rewritten = rewritten[len(prefix):].strip()
    rewritten = rewritten.strip("「」『』\"'")
    return rewritten


def _retrieve_with_dedup(retriever, queries, k):
    """對多組查詢進行檢索並去重，保留排序靠前的結果"""
    seen_contents = set()
    unique_docs = []
    for q in queries:
        docs = retriever.invoke(q)
        for doc in docs:
            content_key = doc.page_content.strip()
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_docs.append(doc)
    return unique_docs[:k]


def _rerank(reranker, question, docs, top_k=3):
    """使用 CrossEncoder 對檢索結果重新排序"""
    if not docs:
        return docs
    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]


# ============================================================
# BM25 Hybrid Search
# ============================================================
def _tokenize_chinese(text):
    """用 jieba 對中文文本進行分詞，過濾空白和標點"""
    tokens = jieba.lcut(text)
    # 過濾空白、標點、單字元符號
    return [t for t in tokens if t.strip() and len(t.strip()) > 0
            and not all(c in '，。、；：「」（）！？\n\r\t ' for c in t)]


def build_bm25_index(vector_db):
    """從 ChromaDB 取出所有文件，建立 BM25 索引"""
    collection = vector_db._collection
    all_data = collection.get(include=["documents", "metadatas"])

    docs_with_meta = []
    tokenized_corpus = []
    for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
        docs_with_meta.append({"page_content": doc_text, "metadata": meta or {}})
        tokenized_corpus.append(_tokenize_chinese(doc_text))

    bm25_index = BM25Okapi(tokenized_corpus)
    print(f"📚 BM25 索引建立完成（{len(tokenized_corpus)} 個 chunks）")
    return bm25_index, docs_with_meta


def _bm25_search(bm25_index, docs_with_meta, query, top_k=9):
    """用 BM25 對查詢進行關鍵字檢索"""
    from langchain_core.documents import Document
    tokenized_query = _tokenize_chinese(query)
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # 只保留有匹配的結果
            doc_info = docs_with_meta[idx]
            doc = Document(
                page_content=doc_info["page_content"],
                metadata=doc_info["metadata"]
            )
            results.append((doc, scores[idx]))
    return results


def _hyde_generate(llm, question):
    """HyDE: 讓 LLM 產生假設性的規章回答，用於改善 vector search 的語義匹配"""
    hyde_prompt = (
        "你是公司內部規章查詢系統。請根據以下員工提問，假設你知道答案，"
        "用正式的公司規章語氣寫出一段可能的回答（約 50-100 字）。\n"
        "即使你不確定答案，也請用合理的規章格式和用語撰寫。\n"
        "必須使用繁體中文，不要加任何前綴或解釋。\n\n"
        f"員工提問：'{question}'"
    )
    hypothetical = llm.invoke(hyde_prompt).strip()
    # 清洗：移除可能的前綴
    for prefix in ["回答：", "答：", "根據公司規章，"]:
        if hypothetical.startswith(prefix):
            hypothetical = hypothetical[len(prefix):].strip()
    return hypothetical


def _hybrid_retrieve(retriever, bm25_index, docs_with_meta, query, fetch_k=9, rrf_k=60,
                     vector_query=None):
    """
    Hybrid Search: Vector + BM25，使用 Reciprocal Rank Fusion (RRF) 合併
    RRF score = sum(1 / (rrf_k + rank_i)) for each retrieval source
    vector_query: 若提供，vector search 用此查詢（HyDE 場景），BM25 仍用 query
    """
    from langchain_core.documents import Document

    # Vector search（可用不同的查詢，如 HyDE 假設性回答）
    vec_q = vector_query if vector_query else query
    original_k = retriever.search_kwargs.get("k", fetch_k)
    retriever.search_kwargs["k"] = fetch_k
    vector_docs = retriever.invoke(vec_q)
    retriever.search_kwargs["k"] = original_k

    # BM25 search（始終用原始查詢，關鍵字匹配更適合）
    bm25_results = _bm25_search(bm25_index, docs_with_meta, query, top_k=fetch_k)

    # RRF 合併：用 page_content 作為去重 key
    rrf_scores = {}  # content -> (score, doc)

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content.strip()
        score = 1.0 / (rrf_k + rank + 1)
        if key in rrf_scores:
            rrf_scores[key] = (rrf_scores[key][0] + score, rrf_scores[key][1])
        else:
            rrf_scores[key] = (score, doc)

    for rank, (doc, _bm25_score) in enumerate(bm25_results):
        key = doc.page_content.strip()
        score = 1.0 / (rrf_k + rank + 1)
        if key in rrf_scores:
            rrf_scores[key] = (rrf_scores[key][0] + score, rrf_scores[key][1])
        else:
            rrf_scores[key] = (score, doc)

    # 按 RRF 分數降序排列
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_results]


def rag_query(retriever, llm, question, rewrite_mode='keywords', k=3, reranker=None,
              bm25_index=None, bm25_docs=None, use_hyde=False, use_fewshot=False,
              use_decompose=False):
    # 如果有 reranker，先多撈一些候選文件再重排
    fetch_k = k * 3 if reranker else k
    use_hybrid = bm25_index is not None and bm25_docs is not None

    # --- Query Decomposition: 複合問題拆分 ---
    if use_decompose:
        sub_queries = _decompose_query(question)
    else:
        sub_queries = [question]

    if use_decompose and len(sub_queries) > 1:
        print(f"  🔀 [decompose] 拆分為 {len(sub_queries)} 個子查詢：{sub_queries}")

    # --- 對每個子查詢進行 rewrite + 檢索，合併結果 ---
    all_retrieved_docs = []
    seen_contents = set()

    for sq in sub_queries:
        # 根據 rewrite_mode 決定搜尋查詢
        if rewrite_mode == 'sentence':
            search_query = _rewrite_sentence(llm, sq)
            if len(sub_queries) == 1:
                print(f"  📝 [sentence] 改寫查詢：{search_query}")
            else:
                print(f"    📝 [sentence] 子查詢改寫：{search_query}")
        elif rewrite_mode == 'dual':
            search_query = _rewrite_sentence(llm, sq)
            if len(sub_queries) == 1:
                print(f"  📝 [dual] 改寫查詢：{search_query}")
            else:
                print(f"    📝 [dual] 子查詢改寫：{search_query}")
        else:
            search_query = _rewrite_keywords(llm, sq)
            if len(sub_queries) == 1:
                print(f"  📝 [keywords] 擴寫關鍵字：{search_query}")
            else:
                print(f"    📝 [keywords] 子查詢關鍵字：{search_query}")

        # HyDE: 產生假設性回答用於 vector search
        hyde_query = None
        if use_hyde:
            hyde_query = _hyde_generate(llm, sq)
            if len(sub_queries) == 1:
                print(f"  🔮 [HyDE] 假設性回答：{hyde_query[:80]}...")
            else:
                print(f"    🔮 [HyDE] 子查詢假設性回答：{hyde_query[:60]}...")

        # 檢索
        if use_hybrid:
            if len(sub_queries) == 1:
                print(f"  🔀 [hybrid] Vector + BM25 → RRF 合併 (fetch_k={fetch_k})")
            if rewrite_mode == 'dual':
                hybrid_docs_1 = _hybrid_retrieve(retriever, bm25_index, bm25_docs, sq,
                                                 fetch_k=fetch_k, vector_query=hyde_query)
                hybrid_docs_2 = _hybrid_retrieve(retriever, bm25_index, bm25_docs, search_query,
                                                 fetch_k=fetch_k, vector_query=hyde_query)
                seen_local = set()
                sub_docs = []
                for doc in hybrid_docs_1 + hybrid_docs_2:
                    key = doc.page_content.strip()
                    if key not in seen_local:
                        seen_local.add(key)
                        sub_docs.append(doc)
                sub_docs = sub_docs[:fetch_k]
            else:
                sub_docs = _hybrid_retrieve(retriever, bm25_index, bm25_docs, search_query,
                                            fetch_k=fetch_k, vector_query=hyde_query)
                if not sub_docs:
                    sub_docs = _hybrid_retrieve(retriever, bm25_index, bm25_docs, sq,
                                                fetch_k=fetch_k, vector_query=hyde_query)
            sub_docs = sub_docs[:fetch_k]
        else:
            original_k = retriever.search_kwargs.get("k", k)
            retriever.search_kwargs["k"] = fetch_k

            vec_q = hyde_query if hyde_query else search_query
            if rewrite_mode == 'dual':
                sub_docs = _retrieve_with_dedup(retriever, [sq, vec_q], fetch_k)
            else:
                sub_docs = retriever.invoke(vec_q)
                if not sub_docs:
                    sub_docs = retriever.invoke(sq)

            retriever.search_kwargs["k"] = original_k

        # 合併去重
        for doc in sub_docs:
            key = doc.page_content.strip()
            if key not in seen_contents:
                seen_contents.add(key)
                all_retrieved_docs.append(doc)

    retrieved_docs = all_retrieved_docs

    # --- Reranker 重排（如果有提供）---
    if reranker is not None:
        print(f"  🔄 [reranker] 重排 {len(retrieved_docs)} → top {k}")
        retrieved_docs = _rerank(reranker, question, retrieved_docs, top_k=k)
    else:
        retrieved_docs = retrieved_docs[:k]

    context = "\n\n".join([d.page_content for d in retrieved_docs])

    if use_fewshot:
        prompt = f"""你是公司內部規章查詢系統。

【回答規則】
1. 仔細閱讀下方【參考資料】，從中找出與問題相關的條文，直接引用作答。
2. 回答必須簡潔扼要，直接給出答案，不要加客套話或問候語。
3. 只有當參考資料中「完全沒有」任何相關內容時，才回答「規章未說明」。
4. 嚴禁添加參考資料中沒有提到的內容，不要自行編造數字或規定。
5. 回答必須使用繁體中文。

以下是三個回答範例，請嚴格參照此格式與風格作答：

【範例一】
參考資料：【採購與固定資產管理作業程序 > 第二章 採購流程與核准權限 > 第 4 條 簽核權責】採購金額 > 200,000 元：一律須呈報總經理（CEO）核准。
員工提問：採購二十五萬的設備需要誰簽核？
回答：根據採購作業程序第4條，採購金額超過200,000元須呈報總經理（CEO）核准。

【範例二】
參考資料：【員工請假管理辦法 > 第二章 假別規定 > 第 3 條 事假】事假應於三日前提出申請，每年以 14 日為限。
員工提問：員工可以帶寵物上班嗎？
回答：規章未說明。參考資料中僅涉及請假規定，未提及攜帶寵物之相關規範。

【範例三】
參考資料：【工時與考勤管理制度 > 第二章 出勤打卡與異常處理 > 第 4 條 遲到、早退與緩衝期機制】打卡時間超過 09:31 且未達 10:00 者視為遲到，當日薪資將直接扣發半小時之基準時薪。每月系統提供給所有員工「2 次、每次 5 分鐘內」之容錯寬限期。自第 3 次微幅遲到起，將嚴格執行前項遲到扣薪規定。
員工提問：這個月已經遲到兩次了，下次遲到會怎樣？
回答：根據工時與考勤管理制度第4條，每月有2次、每次5分鐘內的寬限期。自第3次微幅遲到起將嚴格執行扣薪規定，超過09:31即扣發半小時基準時薪。

【參考資料】
{context}

【員工提問】
{question}

【你的回答】
"""
    else:
        prompt = f"""你是公司內部規章查詢系統。

【回答規則】
1. 仔細閱讀下方【參考資料】，從中找出與問題相關的條文，直接引用作答。
2. 回答必須簡潔扼要，直接給出答案，不要加客套話或問候語。
3. 只有當參考資料中「完全沒有」任何相關內容時，才回答「規章未說明」。
4. 嚴禁添加參考資料中沒有提到的內容，不要自行編造數字或規定。
5. 回答必須使用繁體中文。

【參考資料】
{context}

【員工提問】
{question}

【你的回答】
"""
    answer = llm.invoke(prompt)
    return answer, context

def generate_answers(test_cases, retriever, llm, rewrite_mode='keywords', k=3, reranker=None,
                     bm25_index=None, bm25_docs=None, use_hyde=False, use_fewshot=False,
                     use_decompose=False):
    reranker_label = "+ reranker" if reranker else ""
    hybrid_label = "+ hybrid" if bm25_index else ""
    hyde_label = "+ HyDE" if use_hyde else ""
    fewshot_label = "+ few-shot" if use_fewshot else ""
    decompose_label = "+ decompose" if use_decompose else ""
    print(f"🤖 Step 1: 正在讓本地 RAG 模型進行作答 (rewrite={rewrite_mode} {reranker_label} {hybrid_label} {hyde_label} {fewshot_label} {decompose_label})...")
    results = []

    for i, case in enumerate(test_cases):
        q = case["question"]
        print(f"  [{i+1}/{len(test_cases)}] {q}")
        try:
            answer, context = rag_query(retriever, llm, q, rewrite_mode=rewrite_mode, k=k,
                                        reranker=reranker, bm25_index=bm25_index, bm25_docs=bm25_docs,
                                        use_hyde=use_hyde, use_fewshot=use_fewshot,
                                        use_decompose=use_decompose)
            results.append({
                "question": q,
                "ground_truth": case["ground_truth"],
                "answer": answer,
                "context": context
            })
        except Exception as e:
            print(f"  ❌ 作答失敗: {e}")
            results.append({
                "question": q,
                "ground_truth": case["ground_truth"],
                "answer": f"[ERROR] {e}",
                "context": ""
            })

    print(f"✅ Step 1 完成：共 {len(results)} 筆作答結果\n")
    return results

# ============================================================
# Step 2: Evaluate with Gemini
# ============================================================
def load_regulations(lib_dir="../01_docs_library"):
    regulations = []
    for filepath in glob.glob(os.path.join(lib_dir, "*.md")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            filename = os.path.basename(filepath)
            regulations.append(f"--- File: {filename} ---\n{content}\n")
    return "\n".join(regulations)

def evaluate_with_claude(client, regulations, case):
    prompt = f"""You are a strict and professional evaluator for a RAG (Retrieval-Augmented Generation) system.
You will evaluate the system's performance on a specific question based on company regulations.

### Company Regulations:
{regulations}

### Evaluation Data:
Question: {case['question']}
Ground Truth: {case['ground_truth']}
Retrieved Context by RAG: {case.get('context', '')}
Model's Answer: {case['answer']}

### Instructions:
Evaluate the model's answer based on the retrieved context and ground truth. Output 4 metric scores (0.0 to 1.0) and a brief reasoning in traditional Chinese.
Be extremely strict. If the answer contradicts the regulations or ground truth, scores should be low.

You MUST respond with ONLY a valid JSON object in the following format, no other text:
{{
  "context_precision": <float 0.0-1.0>,
  "context_recall": <float 0.0-1.0>,
  "faithfulness": <float 0.0-1.0>,
  "answer_relevancy": <float 0.0-1.0>,
  "reasoning": "<brief explanation in traditional Chinese>"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def evaluate_answers(answer_results):
    load_dotenv(find_dotenv())
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ 錯誤：找不到 ANTHROPIC_API_KEY，請確認 .env 檔案設定！")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    print("📖 正在載入公司規章...")
    regulations_text = load_regulations()

    print("👨‍🏫 Step 2: 正在呼叫 Claude Sonnet 進行嚴格閱卷...")
    evaluated_results = []

    for i, case in enumerate(answer_results):
        print(f"  [{i+1}/{len(answer_results)}] {case['question']}")
        try:
            result_json = evaluate_with_claude(client, regulations_text, case)
            score_data = json.loads(result_json)
            combined = {**case, **score_data}
            evaluated_results.append(combined)
            print(f"      → {score_data.get('reasoning', '')[:60]}...")
        except Exception as e:
            print(f"  ❌ 評估失敗: {e}")

    if not evaluated_results:
        print("❌ 沒有成功的評估結果！")
        return None

    print(f"✅ Step 2 完成：共 {len(evaluated_results)} 筆評分結果\n")
    return pd.DataFrame(evaluated_results)

# ============================================================
# Step 3: Visualization
# ============================================================
def generate_visual_report(df, output_path="rag_evaluation_report.png", run_label="Baseline"):
    print(f"📊 正在生成視覺化圖表 ({output_path})...")
    metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    metric_labels = ['Context\nPrecision', 'Context\nRecall', 'Faithfulness', 'Answer\nRelevancy']

    avg_scores = [df[m].mean() for m in metrics]
    n = len(df)

    palette = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444']
    bg_color = '#0f172a'
    card_color = '#1e293b'
    text_color = '#f8fafc'
    sub_text = '#94a3b8'
    grid_color = '#334155'

    fig = plt.figure(figsize=(20, 12), facecolor=bg_color)
    fig.suptitle(f'RAG Evaluation Report  —  {run_label}  ({n} Questions)',
                 fontsize=20, fontweight='bold', color=text_color, y=0.97)

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30,
                          left=0.06, right=0.96, top=0.90, bottom=0.07)

    # Panel 1: Radar Chart
    ax_radar = fig.add_subplot(gs[0, 0], polar=True, facecolor=card_color)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    scores = avg_scores + [avg_scores[0]]
    angles_closed = angles + [angles[0]]

    ax_radar.fill(angles_closed, scores, color='#3b82f6', alpha=0.25)
    ax_radar.plot(angles_closed, scores, color='#60a5fa', linewidth=2.5, marker='o', markersize=7)
    for a, s in zip(angles, avg_scores):
        ax_radar.text(a, s + 0.08, f'{s:.2f}', ha='center', va='bottom',
                      fontsize=11, fontweight='bold', color='#60a5fa')
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(metric_labels, fontsize=10, color=text_color)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color=sub_text)
    ax_radar.spines['polar'].set_color(grid_color)
    ax_radar.grid(color=grid_color, linewidth=0.5)
    ax_radar.set_title('Overall Averages', fontsize=13, fontweight='bold',
                       color=text_color, pad=18)

    # Panel 2: Violin + Strip
    ax_violin = fig.add_subplot(gs[0, 1], facecolor=card_color)
    parts = ax_violin.violinplot(
        [df[m].values for m in metrics],
        positions=range(len(metrics)), showmeans=True, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette[i])
        pc.set_alpha(0.35)
    parts['cmeans'].set_color('#fbbf24')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('#f8fafc')
    parts['cmedians'].set_linewidth(1.5)

    for i, m in enumerate(metrics):
        vals = df[m].values
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
        ax_violin.scatter(np.full_like(vals, i) + jitter, vals,
                          color=palette[i], alpha=0.5, s=14, edgecolors='none', zorder=3)

    ax_violin.set_xticks(range(len(metrics)))
    ax_violin.set_xticklabels(metric_labels, fontsize=10, color=text_color)
    ax_violin.set_ylim(-0.05, 1.15)
    ax_violin.set_ylabel('Score', fontsize=11, color=text_color)
    ax_violin.tick_params(colors=sub_text)
    ax_violin.set_facecolor(card_color)
    for spine in ax_violin.spines.values():
        spine.set_color(grid_color)
    ax_violin.grid(axis='y', color=grid_color, linewidth=0.5, alpha=0.5)
    ax_violin.set_title('Score Distribution (Violin + Strip)', fontsize=13,
                        fontweight='bold', color=text_color, pad=12)

    # Panel 3: Summary Table
    ax_table = fig.add_subplot(gs[0, 2], facecolor=card_color)
    ax_table.axis('off')
    stats_data = []
    for m, label in zip(metrics, ['Ctx Prec', 'Ctx Recall', 'Faithful', 'Ans Rel']):
        vals = df[m]
        stats_data.append([
            label,
            f'{vals.mean():.3f}',
            f'{vals.median():.3f}',
            f'{vals.std():.3f}',
            f'{vals.min():.2f}',
            f'{vals.max():.2f}',
            f'{(vals >= 0.8).sum()}/{n}'
        ])
    col_labels = ['Metric', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Pass\n(>=0.8)']
    table = ax_table.table(cellText=stats_data, colLabels=col_labels,
                           loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(grid_color)
        if row == 0:
            cell.set_facecolor('#334155')
            cell.set_text_props(color=text_color, fontweight='bold')
        else:
            cell.set_facecolor(card_color)
            cell.set_text_props(color=text_color)
    ax_table.set_title('Summary Statistics', fontsize=13, fontweight='bold',
                       color=text_color, pad=12)

    # Panel 4: Heatmap (full width)
    ax_heat = fig.add_subplot(gs[1, :], facecolor=card_color)
    df_plot = df.copy()
    df_plot['avg_score'] = df_plot[metrics].mean(axis=1)
    df_plot = df_plot.sort_values('avg_score', ascending=True).reset_index(drop=True)

    heat_data = df_plot[metrics].values.T
    im = ax_heat.imshow(heat_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                        interpolation='nearest')
    ax_heat.set_yticks(range(len(metrics)))
    ax_heat.set_yticklabels(metric_labels, fontsize=10, color=text_color)

    tick_step = max(1, n // 20)
    xtick_pos = list(range(0, n, tick_step))
    ax_heat.set_xticks(xtick_pos)
    ax_heat.set_xticklabels([str(i + 1) for i in xtick_pos], fontsize=8, color=sub_text)
    ax_heat.set_xlabel('Questions (sorted by average score, low → high)', fontsize=11, color=text_color)
    ax_heat.tick_params(colors=sub_text)
    for spine in ax_heat.spines.values():
        spine.set_color(grid_color)
    ax_heat.set_title(f'Per-Question Score Heatmap (all {n} questions)', fontsize=13,
                      fontweight='bold', color=text_color, pad=12)

    cbar = fig.colorbar(im, ax=ax_heat, orientation='vertical', fraction=0.012, pad=0.015)
    cbar.set_label('Score', color=text_color, fontsize=10)
    cbar.ax.tick_params(colors=sub_text)
    cbar.outline.set_edgecolor(grid_color)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"✅ 視覺化圖表已儲存：{output_path}")

# ============================================================
# Single Run Pipeline
# ============================================================
def run_single(test_cases, k, run_label, db_version='v2', rewrite_mode='keywords'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 啟動硬體監控
    hw_monitor = HardwareMonitor(interval=5)
    hw_monitor.start()
    run_start = time.time()

    retriever, llm = init_rag(k=k, db_version=db_version)

    inference_start = time.time()
    answer_results = generate_answers(test_cases, retriever, llm, rewrite_mode=rewrite_mode, k=k)
    inference_duration = time.time() - inference_start

    eval_start = time.time()
    df = evaluate_answers(answer_results)
    eval_duration = time.time() - eval_start

    total_duration = time.time() - run_start
    hw_monitor.stop()

    if df is None:
        return None

    # Print score summary
    print(f"📊 【{run_label} 評估成績單】")
    metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    display_df = df[['question'] + metrics]
    print(display_df.to_markdown(index=False))

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HARDWARE_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, f"{run_label}_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 評分數據已儲存：{csv_path}")

    # Save model answers as JSON for cross-round comparison
    answers_json = []
    for r in answer_results:
        answers_json.append({
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "answer": r["answer"],
            "context": r["context"]
        })
    json_path = os.path.join(RESULTS_DIR, f"{run_label}_{timestamp}_answers.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(answers_json, f, ensure_ascii=False, indent=2)
    print(f"💾 模型回答已儲存：{json_path}")

    png_path = os.path.join(RESULTS_DIR, f"{run_label}_{timestamp}.png")
    generate_visual_report(df, output_path=png_path, run_label=run_label)

    # Save hardware report
    hw_summary = hw_monitor.summary()
    hw_report = {
        "run_label": run_label,
        "timestamp": timestamp,
        "config": {
            "db_version": db_version,
            "k": k,
            "rewrite_mode": rewrite_mode,
            "num_questions": len(test_cases)
        },
        "timing": {
            "total_seconds": round(total_duration, 1),
            "inference_seconds": round(inference_duration, 1),
            "evaluation_seconds": round(eval_duration, 1),
            "avg_inference_per_question": round(inference_duration / len(test_cases), 2)
        },
        "hardware": hw_summary,
        "samples": hw_monitor.samples
    }
    hw_path = os.path.join(HARDWARE_DIR, f"{run_label}_{timestamp}_hardware.json")
    with open(hw_path, "w", encoding="utf-8") as f:
        json.dump(hw_report, f, ensure_ascii=False, indent=2)
    print(f"💾 硬體監控報告已儲存：{hw_path}")

    # Print hardware summary
    print(f"\n⚙️  【硬體耗用摘要】")
    print(f"  總耗時：{total_duration:.0f}s（推論 {inference_duration:.0f}s + 評測 {eval_duration:.0f}s）")
    print(f"  平均每題推論：{inference_duration / len(test_cases):.1f}s")
    if "gpu_util_avg" in hw_summary:
        print(f"  GPU 使用率：平均 {hw_summary['gpu_util_avg']}% / 峰值 {hw_summary['gpu_util_max']}%")
    if "gpu_mem_used_max_mb" in hw_summary:
        print(f"  GPU 記憶體：平均 {hw_summary['gpu_mem_used_avg_mb']}MB / 峰值 {hw_summary['gpu_mem_used_max_mb']}MB (共 {hw_summary.get('gpu_mem_total_mb', '?')}MB)")
    if "ram_used_max_mb" in hw_summary:
        print(f"  系統記憶體：平均 {hw_summary['ram_used_avg_mb']}MB / 峰值 {hw_summary['ram_used_max_mb']}MB (共 {hw_summary.get('ram_total_mb', '?')}MB)")

    return {
        'run_label': run_label,
        'k': k,
        'csv_path': csv_path,
        'png_path': png_path,
        'avg': {m: df[m].mean() for m in metrics},
        'timing': hw_report['timing'],
        'hardware': hw_summary
    }

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    parser.add_argument('--k', type=int, nargs='+', default=[7],
                        help='Retriever k values to test (e.g. --k 5 7 10)')
    parser.add_argument('--db', type=str, default='v2', choices=['v1', 'v2'],
                        help='Vector DB version (v1=chunk500, v2=MarkdownHeader)')
    parser.add_argument('--rewrite', type=str, nargs='+', default=['keywords'],
                        choices=['keywords', 'sentence', 'dual'],
                        help='Query rewrite strategy (keywords=原始關鍵字, sentence=完整書面語改寫, dual=雙路檢索合併)')
    parser.add_argument('--judge', type=str, default='claude',
                        help='評測模型標籤，用於檔名識別 (e.g. claude, gemini)')
    parser.add_argument('--tag', type=str, default=None,
                        help='額外標記，附加在檔名中用於識別特殊改動 (e.g. strict-prompt, reranker)')
    args = parser.parse_args()

    if not os.path.exists(DATASET_FILE):
        print(f"❌ 找不到 {DATASET_FILE}！")
        return
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"📋 載入 {len(test_cases)} 題測試資料")
    print(f"🔬 預計測試 k 值：{args.k}, DB 版本：{args.db}, Rewrite 策略：{args.rewrite}\n")
    print("=" * 60)

    all_results = []
    for rewrite_mode in args.rewrite:
        for k in args.k:
            tag_suffix = f"_{args.tag}" if args.tag else ""
            run_label = f"db{args.db}_k{k}_rw-{rewrite_mode}_j-{args.judge}{tag_suffix}"
            print(f"\n{'=' * 60}")
            print(f"  開始測試：k = {k}, db = {args.db}, rewrite = {rewrite_mode}")
            print(f"{'=' * 60}\n")
            result = run_single(test_cases, k, run_label, db_version=args.db, rewrite_mode=rewrite_mode)
            if result:
                result['rewrite_mode'] = rewrite_mode
                all_results.append(result)

    # Print comparison summary
    if len(all_results) > 1:
        metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
        print(f"\n{'=' * 80}")
        print("📊 【跨策略比較總表】")
        print(f"{'=' * 80}")
        header = f"{'config':<25}" + "".join(f"{m:<18}" for m in metrics) + f"{'推論(s)':<10}{'GPU峰值(MB)':<12}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            label = f"k={r['k']} rw={r.get('rewrite_mode', 'keywords')}"
            row = f"{label:<25}" + "".join(f"{r['avg'][m]:<18.4f}" for m in metrics)
            row += f"{r.get('timing', {}).get('inference_seconds', '-'):<10}"
            row += f"{r.get('hardware', {}).get('gpu_mem_used_max_mb', '-'):<12}"
            print(row)

if __name__ == "__main__":
    main()
