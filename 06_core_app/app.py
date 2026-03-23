import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # 修改1：換成 HuggingFace 介面
from langchain_community.llms import Ollama

app = FastAPI()

# 絕對遵循不連外網的安全限制
# 假設 app.py 放在專案根目錄，指向 04_knowledge_engine/vector_db
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "04_knowledge_engine", "vector_db")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# 1. 載入我們稍早用 BGE-M3 生產好的本地向量資料庫
print("📦 正在連線至本地 RAG ChromaDB 規章資料庫...")
# 修改2：改用業界最強中文語義模型 bge-m3，並直接調用 L4 GPU
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'}
)
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# 讓檢索器每次最多吐出「最相關」的前三段條文
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 2. 勾串本地 Ollama 的 Llama 3 大語言模型 (temperature = 0.0)
print("🧠 正在與 Ollama (Llama 3 模型) 進行握手通訊...")
# 修改3：確保調用的是 llama3
llm = Ollama(model="llama3", temperature=0.0)

# 建立 FastAPI 前端接收模型
class ChatRequest(BaseModel):
    question: str

# 將 static 目錄作為靜態資源伺服出來
# 確保 static 資料夾存在，否則 FastAPI 會報錯
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    print(f"⚠️ 警告：找不到靜態網頁資料夾 {STATIC_DIR}")

@app.get("/")
def read_index():
    # 當使用者連線首頁時，直接吐出我們絕美的毛玻璃特效前端介面
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>網頁前端建置中... (找不到 index.html)</h1>")

@app.post("/api/chat")
async def chat_with_rag(req: ChatRequest):
    question = req.question
    
    # --- 【Query Rewrite 提詞擴寫機制】 ---
    print(f"\n收到原始問題：{question}")
    rewrite_prompt = (
        "請將以下員工口語問題，轉換為 3 到 5 個用來搜尋公司規章的『正式關鍵字』。\n"
        "【嚴格規定】：\n"
        "1. 絕對只能使用「繁體中文」輸出，禁止出現任何英文。\n"
        "2. 關鍵字之間請用空白鍵隔開即可，絕對不要使用項目符號(* 或 -)。\n"
        "3. 不要加入任何解釋、問候或開場白。\n"
        f"原始問題：'{question}'"
    )
    expanded_query = llm.invoke(rewrite_prompt).strip()
    
    # 清洗 expanded_query，只取關鍵詞片段，避免 Llama 廢話干擾搜尋
    if ":" in expanded_query:
        expanded_query = expanded_query.split(":")[-1].strip()
    for noise in ["關鍵字", "正式", "搜尋", "是："]:
        expanded_query = expanded_query.replace(noise, "")
        
    print(f"🚀 Llama 3 擴寫後的搜尋關鍵字：{expanded_query.strip()}")
    
    # 步驟一：拿「清洗後」的關鍵字去尋找相關規章條文
    retrieved_docs = retriever.invoke(expanded_query.strip())
    
    # 雙線備援機制：如果擴寫搜尋沒抓到資料，退回原始提問再搜一次
    if not retrieved_docs or len(retrieved_docs) == 0:
        print("⚠️ 擴寫檢索失敗，啟動備援機制：改用原始提問重新檢索...")
        retrieved_docs = retriever.invoke(question)
        
    print(f"📦 檢索完成，共找到 {len(retrieved_docs)} 段相關條文。")
    context = "\n\n".join([d.page_content for d in retrieved_docs])
    
    # 步驟二：拼裝為給 AI 閱讀的完美 Prompt
    prompt = f"""你是公司資深的「機房維運主任兼人資副理」。
請你「嚴格」根據以下提供的【公司規章內容】來回答員工提出的問題。
如果參考規章中沒有提到該問題的答案，請直接回答「根據目前系統內的規章，無法找到相關規定」。
請用繁體中文回答，口吻專業且簡潔。

【公司規章內容摘錄如下】：
{context}

【員工提出的問題】：
{question}

請根據上述指引提供您的專業回覆：
"""

    # 步驟三：讓 AI 思考並產生回答
    response_text = llm.invoke(prompt)
    
    return {
        "answer": response_text,
        "context_used": context
    }

if __name__ == "__main__":
    print("🚀 啟動 Secure Local RAG 伺服器...")
    # 修改4：將 host 改為 0.0.0.0，允許外部透過 GCP 外部 IP 連線
    print("👉 請打開瀏覽器前往: http://<你的GCP外部IP>:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)