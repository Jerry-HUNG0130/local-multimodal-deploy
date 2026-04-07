"""
RAG 伺服器 — R12d 配置（歷史最佳，Qwen2.5 + Few-Shot v3）
配置：v3 + sentence rewrite + HyDE + BM25 Hybrid (RRF) + Reranker + Qwen2.5 + Few-Shot v3
最佳成績：CP=0.662, CR=0.615, Faith=0.856, AR=0.737, 0 題 Faith=0
"""
import os
import time
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

app = FastAPI()

# ============================================================
# 路徑設定
# ============================================================
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "04_knowledge_engine", "vector_db_v3")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ============================================================
# 初始化元件
# ============================================================
print("📦 正在載入 RAG 引擎 (R12d 配置 — Qwen2.5 + Few-Shot v3)...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'}
)
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 9})

llm = Ollama(model="qwen2.5:7b-instruct-q4_K_M", temperature=0.0)

# BM25 索引
print("📚 建立 BM25 索引...")
collection = vector_db._collection
all_data = collection.get(include=["documents", "metadatas"])
bm25_docs = []
bm25_corpus = []
for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
    bm25_docs.append({"page_content": doc_text, "metadata": meta or {}})
    tokens = [t for t in jieba.lcut(doc_text)
              if t.strip() and not all(c in '，。、；：「」（）！？\n\r\t ' for c in t)]
    bm25_corpus.append(tokens)
bm25_index = BM25Okapi(bm25_corpus)
print(f"📚 BM25 索引就緒（{len(bm25_corpus)} chunks）")

# Reranker
print("🔄 載入 Reranker...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda")

print("✅ R12d RAG 引擎就緒（Qwen2.5 + Few-Shot v3）\n")


# ============================================================
# 核心函數
# ============================================================
def rewrite_sentence(question):
    prompt = (
        "你是公司規章檢索系統的查詢優化器。\n"
        "請將以下員工的口語問題，改寫為一句「完整的正式書面語句子」，用於搜尋公司內部規章。\n"
        "【嚴格規定】：\n"
        "1. 只輸出改寫後的一句話，不要加任何解釋或前綴。\n"
        "2. 必須使用繁體中文。\n"
        "3. 保留原始問題的完整語意，用正式的公司規章用語重新表達。\n"
        "4. 不要拆成關鍵字，必須是一句完整、通順的句子。\n"
        f"原始問題：'{question}'"
    )
    rewritten = llm.invoke(prompt).strip()
    for prefix in ["改寫後：", "改寫：", "查詢：", "問題："]:
        if rewritten.startswith(prefix):
            rewritten = rewritten[len(prefix):].strip()
    return rewritten.strip("「」『』\"'")


def hyde_generate(question):
    prompt = (
        "你是公司內部規章查詢系統。請根據以下員工提問，假設你知道答案，"
        "用正式的公司規章語氣寫出一段可能的回答（約 50-100 字）。\n"
        "即使你不確定答案，也請用合理的規章格式和用語撰寫。\n"
        "必須使用繁體中文，不要加任何前綴或解釋。\n\n"
        f"員工提問：'{question}'"
    )
    hypothetical = llm.invoke(prompt).strip()
    for prefix in ["回答：", "答：", "根據公司規章，"]:
        if hypothetical.startswith(prefix):
            hypothetical = hypothetical[len(prefix):].strip()
    return hypothetical


def bm25_search(query, top_k=9):
    from langchain_core.documents import Document
    tokens = [t for t in jieba.lcut(query)
              if t.strip() and not all(c in '，。、；：「」（）！？\n\r\t ' for c in t)]
    scores = bm25_index.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [Document(page_content=bm25_docs[i]["page_content"], metadata=bm25_docs[i]["metadata"])
            for i in top_indices if scores[i] > 0]


def hybrid_retrieve(bm25_query, vector_query=None, fetch_k=9):
    vec_q = vector_query if vector_query else bm25_query

    original_k = retriever.search_kwargs.get("k", fetch_k)
    retriever.search_kwargs["k"] = fetch_k
    vector_results = retriever.invoke(vec_q)
    retriever.search_kwargs["k"] = original_k

    bm25_results = bm25_search(bm25_query, top_k=fetch_k)

    rrf_k = 60
    rrf_scores = {}
    for rank, doc in enumerate(vector_results):
        key = doc.page_content.strip()
        score = 1.0 / (rrf_k + rank + 1)
        rrf_scores[key] = (rrf_scores.get(key, (0, doc))[0] + score, doc)
    for rank, doc in enumerate(bm25_results):
        key = doc.page_content.strip()
        score = 1.0 / (rrf_k + rank + 1)
        if key in rrf_scores:
            rrf_scores[key] = (rrf_scores[key][0] + score, rrf_scores[key][1])
        else:
            rrf_scores[key] = (score, doc)

    sorted_results = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_results[:fetch_k]]


def build_prompt(context, question):
    """R12d 的 Few-Shot v3 prompt — 歷史最佳配置"""
    return f"""你是公司內部規章查詢系統。

【回答規則】
1. 仔細閱讀下方【參考資料】，從中找出與問題相關的條文，直接引用作答。
2. 回答必須簡潔扼要，直接給出答案，不要加客套話或問候語。
3. 只有當參考資料中「完全沒有」任何相關內容時，才回答「規章未說明」。
4. 嚴禁添加參考資料中沒有提到的內容，不要自行編造數字或規定。
5. 回答必須使用繁體中文。

【範例一】正確引用條文：
參考資料提到「採購金額 > 200,000 元：一律須呈報總經理（CEO）核准」
員工提問：買一台 25 萬的 GPU，要簽核到哪個層級？
正確回答：採購金額超過 200,000 元，須呈報總經理（CEO）核准。

【範例二】注意排除條款：
參考資料提到「1. 全職工程師皆具備申請資格。2. 第一線機房維運者不得申請。」
員工提問：我是機房 SRE，可以申請嗎？
正確回答：不行。第一線機房維運者因工作性質，不得申請。

【範例三】規章未涵蓋時：
參考資料僅提到加班費與補休轉換規定。
員工提問：春節加班費是幾倍？
正確回答：規章未說明。

【參考資料】
{context}

【員工提問】
{question}

【你的回答】
"""


# ============================================================
# API 路由
# ============================================================
class ChatRequest(BaseModel):
    question: str

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>前端建置中...</h1>")

@app.post("/api/chat")
async def chat_with_rag(req: ChatRequest):
    question = req.question
    start_time = time.time()

    print(f"\n{'='*50}")
    print(f"📝 收到問題：{question}")

    # Step 1: Sentence Rewrite
    search_query = rewrite_sentence(question)
    print(f"📝 [sentence] 改寫：{search_query}")

    # Step 2: HyDE
    hyde_query = hyde_generate(question)
    print(f"🔮 [HyDE] 假答案：{hyde_query[:60]}...")

    # Step 3: Hybrid Retrieve
    retrieved_docs = hybrid_retrieve(bm25_query=search_query, vector_query=hyde_query, fetch_k=9)
    if not retrieved_docs:
        retrieved_docs = hybrid_retrieve(bm25_query=question, vector_query=hyde_query, fetch_k=9)
    print(f"🔀 [hybrid] 候選 {len(retrieved_docs)} 篇")

    # Step 4: Reranker → top 3
    if retrieved_docs:
        pairs = [[question, doc.page_content] for doc in retrieved_docs]
        scores = reranker.predict(pairs)
        scored_docs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
        retrieved_docs = [doc for _, doc in scored_docs[:3]]
    print(f"🔄 [reranker] 精選 top 3")

    context = "\n\n".join([d.page_content for d in retrieved_docs])

    # Step 5: LLM 生成回答（Few-Shot v3 prompt）
    prompt = build_prompt(context, question)
    response_text = llm.invoke(prompt)
    elapsed = time.time() - start_time
    print(f"✅ 回答完成（{elapsed:.1f}s）")

    return JSONResponse(
        content={
            "answer": response_text,
            "context_used": context,
            "elapsed_seconds": round(elapsed, 1)
        },
        media_type="application/json; charset=utf-8"
    )

if __name__ == "__main__":
    print("🚀 啟動 RAG 伺服器（R12d 配置 — Qwen2.5 + Few-Shot v3，歷史最佳）")
    print("👉 http://<GCP外部IP>:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
