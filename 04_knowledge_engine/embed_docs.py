import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # 改用這個
from langchain_community.vectorstores import Chroma

# 1. 路徑設定 (保持你要求的標準結構)
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "01_docs_library")
DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")

def main():
    print("🚀 正在讀取公司規章 Markdown 檔案...")
    md_files = glob.glob(os.path.join(DOCS_DIR, "*.md"))
    
    if not md_files:
        print(f"❌ 錯誤：在 {DOCS_DIR} 找不到任何 .md 檔案！")
        return

    docs = []
    for file_path in md_files:
        # 確保使用 utf-8 讀取繁體中文
        loader = TextLoader(file_path, encoding='utf-8')
        docs.extend(loader.load())
    
    print(f"📄 共讀取 {len(docs)} 份文件，準備進行文字切塊 (Chunking)...")
    
    # 2. 文字切塊：針對中文，chunk_size 500 字左右效果最佳
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    print(f"✂️ 共切分為 {len(split_docs)} 塊內容。")
    
    # 3. 更換為業界標竿的中文嵌入模型 (BGE-M3)
    # 雖然它是中國研發，但在開源界是「純本地運行」，不需連網，不涉及 API 介接安全性
    print("🧠 啟動 BGE-M3 語義檢索模型 (使用 L4 GPU 加速)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'} # 指定使用 GPU
    )
    
    # 4. 建立 ChromaDB 向量庫
    print("💾 正在轉換並儲存至 ChromaDB 本地向量資料庫...")
    db = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print(f"✅ RAG 知識庫建立完成！資料庫位置: {DB_DIR}")

if __name__ == "__main__":
    main()