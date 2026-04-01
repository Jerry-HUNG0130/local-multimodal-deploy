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
# v2: MarkdownHeaderTextSplitter (原始，無標題前綴)
DB_DIR_V2 = os.path.join(os.path.dirname(__file__), "vector_db_v2")
# v3: MarkdownHeaderTextSplitter + 標題前綴
DB_DIR_V3 = os.path.join(os.path.dirname(__file__), "vector_db_v3")
# v4: MarkdownNodeParser + 標題前綴
DB_DIR_V4 = os.path.join(os.path.dirname(__file__), "vector_db_v4")
# v5: MarkdownHeaderTextSplitter + 標題前綴 + Contextual Retrieval (LLM 語義前綴)
DB_DIR_V5 = os.path.join(os.path.dirname(__file__), "vector_db_v5")

HEADERS_TO_SPLIT_ON = [
    ("#",   "document_title"),
    ("##",  "chapter"),
    ("###", "article"),
]


def _build_title_prefix(metadata):
    """從 metadata 組合標題前綴，如【員工請假管理辦法 > 假別規定 > 第6條 公假】"""
    parts = []
    for key in ["document_title", "chapter", "article"]:
        if key in metadata and metadata[key]:
            parts.append(metadata[key].strip())
    if parts:
        return "【" + " > ".join(parts) + "】\n"
    return ""


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


def load_and_split_markdown_v3(docs_dir):
    """v3: MarkdownHeaderTextSplitter + 標題前綴拼接到 page_content"""
    chunks = load_and_split_markdown(docs_dir)
    for chunk in chunks:
        prefix = _build_title_prefix(chunk.metadata)
        chunk.page_content = prefix + chunk.page_content
    return chunks


def load_and_split_markdown_v4(docs_dir):
    """v4: LlamaIndex MarkdownNodeParser → 轉換為 LangChain Document + 標題前綴"""
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.core import Document as LIDocument

    md_files = sorted(glob.glob(os.path.join(docs_dir, "*.md")))
    if not md_files:
        print(f"❌ 在 {docs_dir} 找不到任何 .md 檔案！")
        return []

    parser = MarkdownNodeParser()
    all_chunks = []

    for file_path in md_files:
        filename = os.path.basename(file_path)
        print(f"  📄 {filename}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # LlamaIndex Document → MarkdownNodeParser
        li_doc = LIDocument(text=content, metadata={"source_file": filename})
        nodes = parser.get_nodes_from_documents([li_doc])

        for node in nodes:
            text = node.text.strip()
            # 過濾只有標題沒有內容的中間節點（如 "## 第一章 總則"）
            lines = text.split("\n")
            content_lines = [l for l in lines if not l.startswith("#") and l.strip()]
            if not content_lines:
                continue

            # 從 header_path 解析層級（格式如 /員工請假管理辦法/第二章 假別規定/）
            header_path = node.metadata.get("header_path", "/")
            path_parts = [p for p in header_path.strip("/").split("/") if p]

            # 建立 LangChain Document metadata
            meta = {"source_file": filename}
            if len(path_parts) >= 1:
                meta["document_title"] = path_parts[0]
            if len(path_parts) >= 2:
                meta["chapter"] = path_parts[1]
            if len(path_parts) >= 3:
                meta["article"] = path_parts[2]

            # 組合標題前綴 + 原始文本
            prefix = _build_title_prefix(meta)
            page_content = prefix + text

            lc_doc = Document(page_content=page_content, metadata=meta)
            all_chunks.append(lc_doc)

    return all_chunks


def load_and_split_markdown_v5(docs_dir, llm_model='qwen2.5:7b-instruct-q8_0'):
    """v5: v3 基礎 + Contextual Retrieval（用 LLM 為每個 chunk 產生語義定位說明）

    Anthropic Contextual Retrieval 方法：
    將整份文件作為 context，讓 LLM 為每個 chunk 生成一段簡短的定位描述，
    說明此 chunk 在文件中的角色和關聯，拼接到 chunk 前面後再 embedding。
    """
    from langchain_community.llms import Ollama

    # 先用 v3 的方式切分（含標題前綴）
    chunks = load_and_split_markdown_v3(docs_dir)

    # 讀取完整文件內容（以 source_file 分組）
    full_docs = {}
    for file_path in sorted(glob.glob(os.path.join(docs_dir, "*.md"))):
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            full_docs[filename] = f.read()

    llm = Ollama(model=llm_model, temperature=0.0)
    print(f"\n🧠 Contextual Retrieval: 使用 {llm_model} 為 {len(chunks)} 個 chunks 生成語義前綴...")

    for i, chunk in enumerate(chunks):
        source_file = chunk.metadata.get("source_file", "")
        full_doc = full_docs.get(source_file, "")

        # 截取 chunk 的純文本（去掉 v3 的標題前綴）
        chunk_text = chunk.page_content

        prompt = (
            "<document>\n"
            f"{full_doc}\n"
            "</document>\n\n"
            "以下是上述文件中的一個片段：\n"
            "<chunk>\n"
            f"{chunk_text}\n"
            "</chunk>\n\n"
            "請用 1-2 句繁體中文，簡要說明這個片段在整份文件中的定位與用途。"
            "直接輸出說明，不要加前綴或解釋。"
        )
        try:
            context_desc = llm.invoke(prompt).strip()
            # 清洗：移除可能的前綴
            for prefix in ["說明：", "定位：", "這個片段"]:
                if context_desc.startswith(prefix):
                    context_desc = context_desc[len(prefix):].strip()
            # 將語義描述加到 chunk 最前面
            chunk.page_content = f"[語義定位] {context_desc}\n{chunk.page_content}"
            print(f"  [{i+1}/{len(chunks)}] ✅ {context_desc[:60]}...")
        except Exception as e:
            print(f"  [{i+1}/{len(chunks)}] ❌ LLM 呼叫失敗: {e}")

    return chunks


DB_DIR_MAP = {
    'v1': DB_DIR_V1,
    'v2': DB_DIR_V2,
    'v3': DB_DIR_V3,
    'v4': DB_DIR_V4,
    'v5': DB_DIR_V5,
}


def main():
    parser = argparse.ArgumentParser(description="Build RAG vector database")
    parser.add_argument('--version', type=str, default='v2', choices=['v1', 'v2', 'v3', 'v4', 'v5'],
                        help='v1=RecursiveCharacter(500), v2=MarkdownHeader, v3=MarkdownHeader+標題前綴, v4=NodeParser+標題前綴, v5=v3+Contextual Retrieval')
    parser.add_argument('--llm-model', type=str, default='qwen2.5:7b-instruct-q8_0',
                        help='LLM model for contextual retrieval (v5 only)')
    args = parser.parse_args()

    db_dir = DB_DIR_MAP[args.version]

    print(f"🚀 正在讀取公司規章 Markdown 檔案 (version={args.version})...")

    if args.version == 'v5':
        chunks = load_and_split_markdown_v5(DOCS_DIR, llm_model=args.llm_model)
    elif args.version == 'v4':
        chunks = load_and_split_markdown_v4(DOCS_DIR)
    elif args.version == 'v3':
        chunks = load_and_split_markdown_v3(DOCS_DIR)
    elif args.version == 'v2':
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
