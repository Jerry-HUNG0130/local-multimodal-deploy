import os
import glob
import argparse
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "01_docs_library")

# v1: original chunking (kept for reference)
DB_DIR_V1 = os.path.join(os.path.dirname(__file__), "vector_db")
# v2: MarkdownHeaderTextSplitter
DB_DIR_V2 = os.path.join(os.path.dirname(__file__), "vector_db_v2")

HEADERS_TO_SPLIT_ON = [
    ("#",   "document_title"),
    ("##",  "chapter"),
    ("###", "article"),
]

def load_and_split_markdown(docs_dir):
    md_files = sorted(glob.glob(os.path.join(docs_dir, "*.md")))
    if not md_files:
        print(f"❌ 在 {docs_dir} 找不到任何 .md 檔案！")
        return []

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    all_chunks = []
    for file_path in md_files:
        filename = os.path.basename(file_path)
        print(f"  📄 {filename}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = splitter.split_text(content)

        for chunk in chunks:
            chunk.metadata["source_file"] = filename
            all_chunks.append(chunk)

    return all_chunks

def main():
    parser = argparse.ArgumentParser(description="Build RAG vector database")
    parser.add_argument('--version', type=str, default='v2', choices=['v1', 'v2'],
                        help='v1=RecursiveCharacter(500), v2=MarkdownHeader')
    args = parser.parse_args()

    db_dir = DB_DIR_V1 if args.version == 'v1' else DB_DIR_V2

    print(f"🚀 正在讀取公司規章 Markdown 檔案 (version={args.version})...")

    if args.version == 'v2':
        chunks = load_and_split_markdown(DOCS_DIR)
    else:
        # v1: legacy chunking for reference
        from langchain_community.document_loaders import TextLoader
        docs = []
        for file_path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.md"))):
            loader = TextLoader(file_path, encoding='utf-8')
            docs.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

    print(f"✂️ 共切分為 {len(chunks)} 塊內容。")

    # Preview first 3 chunks
    for i, c in enumerate(chunks[:3]):
        print(f"\n  --- Chunk {i+1} preview ---")
        print(f"  metadata: {c.metadata}")
        print(f"  content:  {c.page_content[:80]}...")

    print(f"\n🧠 啟動 BGE-M3 語義檢索模型 (使用 L4 GPU 加速)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'}
    )

    print(f"💾 正在轉換並儲存至 ChromaDB ({db_dir})...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )
    print(f"✅ RAG 知識庫建立完成！共 {len(chunks)} 個 chunks，位置: {db_dir}")

if __name__ == "__main__":
    main()
