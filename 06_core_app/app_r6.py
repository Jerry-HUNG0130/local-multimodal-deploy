"""
RAG 伺服器 — R6 配置（速度優先，~4s/題）
配置：v3 + sentence rewrite + BM25 Hybrid (RRF) + Reranker
適合：面試現場即時展示
"""
import os
import time
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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
print("📦 正在載入 RAG 引擎 (R6 配置)...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'}
)
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 9})

llm = Ollama(model="llama3", temperature=0.0)

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

print("✅ R6 RAG 引擎就緒\n")


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


def bm25_search(query, top_k=9):
    from langchain_core.documents import Document
    tokens = [t for t in jieba.lcut(query)
              if t.strip() and not all(c in '，。、；：「」（）！？\n\r\t ' for c in t)]
    scores = bm25_index.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [Document(page_content=bm25_docs[i]["page_content"], metadata=bm25_docs[i]["metadata"])
            for i in top_indices if scores[i] > 0]


def hybrid_retrieve(query, fetch_k=9):
    vector_results = retriever.invoke(query)
    bm25_results = bm25_search(query, top_k=fetch_k)

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

    # Step 2: Hybrid Retrieve (Vector + BM25 → RRF)
    retrieved_docs = hybrid_retrieve(search_query, fetch_k=9)
    if not retrieved_docs:
        retrieved_docs = hybrid_retrieve(question, fetch_k=9)
    print(f"🔀 [hybrid] 候選 {len(retrieved_docs)} 篇")

    # Step 3: Reranker → top 3
    if retrieved_docs:
        pairs = [[question, doc.page_content] for doc in retrieved_docs]
        scores = reranker.predict(pairs)
        scored_docs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
        retrieved_docs = [doc for _, doc in scored_docs[:3]]
    print(f"🔄 [reranker] 精選 top 3")

    context = "\n\n".join([d.page_content for d in retrieved_docs])

    # Step 4: LLM 生成回答
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
    response_text = llm.invoke(prompt)
    elapsed = time.time() - start_time
    print(f"✅ 回答完成（{elapsed:.1f}s）")

    return {
        "answer": response_text,
        "context_used": context,
        "elapsed_seconds": round(elapsed, 1)
    }

if __name__ == "__main__":
    print("🚀 啟動 RAG 伺服器（R6 配置 — 速度優先 ~4s/題）")
    print("👉 http://<GCP外部IP>:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
