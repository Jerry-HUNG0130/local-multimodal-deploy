import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 絕對遵循要求：完全使用本地端組件，無任何中國雲端 API 介接
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "01_docs_library")
DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")

def main():
    print("正在讀取公司規章 Markdown 檔案...")
    md_files = glob.glob(os.path.join(DOCS_DIR, "*.md"))
    
    if not md_files:
        print(f"錯誤：在 {DOCS_DIR} 找不到任何 .md 檔案！")
        return

    docs = []
    for file_path in md_files:
        loader = TextLoader(file_path, encoding='utf-8')
        docs.extend(loader.load())
    
    print(f"共讀取 {len(docs)} 份文件，準備進行文字切塊 (Chunking)...")
    
    # 2. 為了讓 AI 能精確找到段落，將長文章切分成小塊
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    print(f"共切分為 {len(split_docs)} 塊內容。")
    
    # 3. 遵守不使用中國開源軟體的守則，改用專為 CPU 及查字典超優化、體積極小的 mxbai-embed-large 嵌入模型
    print("啟動本地 Ollama 向量模型 (mxbai-embed-large)...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # 4. 建立 ChromaDB 向量庫並存檔
    print("正在轉換並儲存至 ChromaDB 本地向量資料庫...")
    # Chroma 預設會自動將資料寫入 persist_directory 中
    db = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print(f"✅ RAG 知識庫建立完成！資料庫已安全存放在: {DB_DIR}")

if __name__ == "__main__":
    main()
