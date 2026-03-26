# Project: Local Multimodal Deploy

## Purpose
Interview portfolio project. Demonstrates the ability to deploy, evaluate, and iteratively improve a fully local multimodal AI system on a GCP VM (L4 GPU).

## What This Project Does
- Deploys an offline RAG (Retrieval-Augmented Generation) system using open-source models
- Uses simulated company regulations (11 HR/IT policy documents in Traditional Chinese) as the knowledge base
- Evaluates model performance with 100 test questions scored by Claude Code（直接讀取評分，不呼叫 API）
- Iteratively improves results through prompt engineering, data formatting, retrieval tuning, etc.
- Each improvement round is saved in `07_evaluation_results/` for side-by-side comparison

## Development Environment
- **Virtual Environment**: This project uses `.venv`. Always activate it before running commands (`source .venv/bin/activate`). NEVER install packages globally — always use `pip install` inside the `.venv` environment.
- **Security**: 安裝 `litellm` 時必須避開 1.82.7 和 1.82.8 版本（已知資安漏洞）。

## Tech Stack
- **LLM**: Llama 3 via Ollama (local, no internet)
- **Embedding**: BAAI/bge-m3 on CUDA (HuggingFace)
- **Vector DB**: ChromaDB (local)
- **Backend**: FastAPI (port 8000)
- **Frontend**: Single-page glassmorphism chat UI
- **Evaluation**: Claude Code 直接讀取 answers.json + 規章文件評分（不呼叫 API）
- **BM25**: rank_bm25 + jieba（中文分詞）
- **Reranker**: BAAI/bge-reranker-v2-m3 (CrossEncoder)
- **Infra**: GCP VM with NVIDIA L4 GPU

## Project Structure
```
01_docs_library/        # 11 company regulation docs (.md, Traditional Chinese)
02_models/              # Model-related files
03_data_simulation/     # Simulated test data
04_knowledge_engine/    # embed_docs.py + ChromaDB vector store
05_vision_engine/       # YOLO-based server room detection (WIP)
06_core_app/            # FastAPI app, evaluation pipeline, frontend
07_evaluation_results/
  ├── scores/           # 每輪測評產出（CSV 評分 + PNG 圖表 + answers JSON 模型回答）
  ├── hardware/         # 硬體監控報告（每次測評自動產出 + 獨立監控 log）
  └── baseline/         # 基線測試（如 embedding 模型切換前的初始結果）
```

## Key Workflow
1. `04_knowledge_engine/embed_docs.py` — vectorize docs into ChromaDB
   - `--version v1`: RecursiveCharacterTextSplitter(chunk_size=500) → `vector_db/`
   - `--version v2`: MarkdownHeaderTextSplitter (by #/##/### structure) → `vector_db_v2/`
   - `--version v3`: MarkdownHeaderTextSplitter + 標題前綴拼接 → `vector_db_v3/`（**目前最佳**）
   - `--version v4`: LlamaIndex MarkdownNodeParser + 標題前綴 → `vector_db_v4/`
   - v2/v3 splits by document structure, producing 76 chunks with metadata (document_title, chapter, article, source_file)
2. `06_core_app/app.py` — 原始 RAG API（keyword rewrite, vector_db_v2）— 已過時，保留作參考
3. `06_core_app/app_r6.py` — **R6 配置 RAG API（速度優先 ~4s/題）**
   - v3 + sentence rewrite + BM25 Hybrid (RRF) + Reranker
   - 適合面試現場即時展示
4. `06_core_app/app_r8.py` — **R8a 配置 RAG API（品質優先 ~8s/題）**
   - v3 + sentence rewrite + HyDE + BM25 Hybrid (RRF) + Reranker
   - 適合品質展示、離線評測
3. `06_core_app/evaluate_pipeline.py` — unified evaluation pipeline
   - Direct RAG call (no need to start FastAPI server)
   - Usage: `python evaluate_pipeline.py --k 3 --db v2 --rewrite sentence --judge claude --tag some-note`
   - Golden dataset: `06_core_app/golden_dataset.json` (100 questions + ground truth)
   - **評測方式：由 Claude Code 直接讀取 answers.json + 01_docs_library/ 規章文件進行評分，不呼叫 Anthropic API。** 使用 `eval_helper.py read` 分批讀取回答，Claude Code 逐題評分後用 `eval_helper.py write` 寫入 CSV，最後用 `eval_helper.py report` 產出圖表。
   - 每次執行產出 4 個檔案：
     - `07_evaluation_results/scores/{run_label}_{timestamp}.csv` — 每題評分數據
     - `07_evaluation_results/scores/{run_label}_{timestamp}.png` — 視覺化圖表
     - `07_evaluation_results/scores/{run_label}_{timestamp}_answers.json` — 模型回答紀錄（含 question, ground_truth, answer, context），用於跨 Round 比對同一題的回答變化
     - `07_evaluation_results/hardware/{run_label}_{timestamp}_hardware.json` — 硬體監控報告（GPU/CPU/RAM 使用率 + 推論耗時），展現成本控制而非一味追求極致表現
   - **改動紀錄**：`07_evaluation_results/CHANGELOG.txt` — 記錄從 Llama3 原始模型起各階段的改動內容、對應測試結果檔名與關鍵數據。**每次進行改動與測試時必須同步更新此檔案。**

### 評測檔案命名規則
格式：`db{版本}_k{值}_rw-{rewrite策略}_j-{評測模型}[_{tag}]_{timestamp}`
- `db`: 向量資料庫版本 (`v1`=chunk500, `v2`=MarkdownHeader)
- `k`: retriever 取回文件數
- `rw`: query rewrite 策略 (`keywords`, `sentence`, `dual`)
- `j`: 評測模型 (`gemini`, `claude`)
- `tag`（可選）：額外標記，用 `--tag` 指定（如 `strict-prompt`, `balanced-prompt`, `reranker`）
- `timestamp`: 執行時間 `YYYYMMDD_HHMMSS`

範例：`dbv2_k3_rw-sentence_j-claude_20260325_140000.csv`

### 歷史評測檔案對照表
以下為 `07_evaluation_results/` 中既有檔案與其對應的改動說明：

**`baseline/`**
| 檔名 | 說明 |
|------|------|
| `baseline_demo.png` | Llama3 + bge-m3 embedding 初始基線測試結果 |

**`hardware/`**
| 檔名 | 說明 |
|------|------|
| `hardware_log.csv` | 獨立硬體監控腳本 (`monitor.sh`) 的採樣記錄，非 pipeline 自動產出 |

**`scores/`**（Round 1~4 均由 Gemini 評測，無 answers.json 和 hardware.json）
| 檔名前綴 | Round | 改動內容 |
|-----------|-------|---------|
| `k5_20260324_031204` | R1 | v1 + k=5, keywords rewrite |
| `k7_20260324_033329` | R1 | v1 + k=7, keywords rewrite |
| `k10_20260324_035502` | R1 | v1 + k=10, keywords rewrite |
| `dbv2_k3_20260324_050250` | R2 | v2 MarkdownHeader + k=3, keywords rewrite |
| `dbv2_k5_20260324_052347` | R2 | v2 + k=5, keywords rewrite |
| `dbv2_k7_20260324_054622` | R2 | v2 + k=7, keywords rewrite |
| `dbv2_k3_rw-sentence_20260324_125442` | R3 | v2 + k=3, sentence rewrite |
| `dbv2_k3_rw-dual_20260324_131809` | R3 | v2 + k=3, dual rewrite |
| `dbv2_k3_rw-sentence_20260324_141350` | R4a | v2 + k=3, sentence rewrite + ultra-strict prompt |
| `dbv2_k3_rw-sentence_20260325_022924` | R4b | v2 + k=3, sentence rewrite + balanced prompt |

> 注意：R1 的檔名缺少 `db` 前綴（當時尚未引入 db 版本參數），均為 v1。R4a/R4b 與 R3 的 `run_label` 相同（差異僅在 prompt 內容），需靠 timestamp 區分。自此之後新增 `--tag` 參數避免此問題。
> 自 Round 5 起，pipeline 自動產出 `_answers.json`（模型回答）和 `_hardware.json`（硬體監控），且評測改用 Claude Sonnet。

## Key Context
- All inference must stay offline (security constraint) — only evaluation calls Claude API externally
- The user's language is Traditional Chinese; code comments are in Chinese
- Focus is on measurable improvement across rounds, not just a single deployment

## Improvement Log

### Round 1: Retriever k-value tuning (2026-03-24)
Tested k=5, k=7, k=10 against baseline k=3.

| Metric             | k=5   | k=7   | k=10  |
|--------------------|-------|-------|-------|
| Context Precision  | 0.374 | 0.340 | 0.365 |
| Context Recall     | 0.458 | 0.440 | 0.545 |
| Faithfulness       | 0.609 | 0.613 | 0.536 |
| Answer Relevancy   | 0.592 | 0.613 | 0.508 |

**Decision: k=7** — best balance across all metrics.
- k=10 improves recall but floods the prompt with noise, causing Faithfulness and Answer Relevancy to drop.
- Context Precision remains low (<0.38) across all k values, indicating the root problem is retrieval quality, not quantity.
- Next steps: improve chunk strategy, add document title metadata, optimize Query Rewrite.

### Round 2: Chunking strategy — MarkdownHeaderTextSplitter (2026-03-24)
Replaced RecursiveCharacterTextSplitter(500) with MarkdownHeaderTextSplitter.

**Document analysis (11 files, 76 subsections):**
- Subsection lengths: min=61, max=632, avg=176, median=151 chars
- 98.7% of subsections are under 400 chars
- All files follow consistent `# Title > ## Chapter > ### Article` structure

**v1 vs v2 vector DB:**
- v1 (`vector_db/`): ~30 chunks, fixed 500-char splits, no metadata, cross-article contamination
- v2 (`vector_db_v2/`): 76 chunks, one per regulation article, with metadata (document_title, chapter, article, source_file)

**Evaluation results (v2, k=3/5/7):**

| Metric             | v2 k=3 | v2 k=5 | v2 k=7 | (v1 k=7 for ref) |
|--------------------|--------|--------|--------|-------------------|
| Context Precision  | 0.344  | 0.332  | 0.307  | 0.340             |
| Context Recall     | 0.348  | 0.418  | 0.418  | 0.440             |
| Faithfulness       | **0.802** | 0.644 | 0.719 | 0.613            |
| Answer Relevancy   | **0.645** | 0.645 | 0.638 | 0.613            |
| Faith Pass(>=0.8)  | **77/100** | 59/100 | 68/100 | 54/100          |

**Key findings:**
- Faithfulness jumped significantly: v2 k=3 (0.802) vs v1 k=7 (0.613) — +31% improvement
- Answer Relevancy also improved: 0.645 vs 0.613 — model gives better answers when context is cleaner
- Context Precision/Recall did NOT improve — the retrieval stage still struggles to find the right chunks
- v2 k=3 is the best config so far: fewer but cleaner chunks → LLM is less confused → higher Faithfulness
- The improvement confirms: v1's 500-char mixed chunks were polluting the LLM prompt with noise

**Decision: v2 + k=3** — Faithfulness is the biggest gain, and k=3 outperforms k=5/7 on v2.
- Next bottleneck: Context Precision/Recall still low (~0.34). The retrieval itself needs improvement.
- Potential next steps: optimize Query Rewrite, or try hybrid search (vector + keyword).

### Round 3: Query Rewrite strategy optimization (2026-03-24)
Analyzed the existing Query Rewrite mechanism and identified a fundamental mismatch: the prompt asks Llama3 to output discrete "keywords", but the downstream search uses vector (semantic) retrieval. Splitting a meaningful question into loose keywords destroys the semantic vector quality.

**Tested two new strategies (both on v2 + k=3):**
- **B: Sentence rewrite (`sentence`)** — Rewrite the user's colloquial question into a single formal, complete sentence using official regulatory language. Preserves semantic integrity for vector search.
- **C: Dual-path retrieval (`dual`)** — Search with BOTH the original question AND the rewritten sentence, then merge and deduplicate results (top k). Aims to maximize recall by combining both perspectives.

**Implementation:** Added `--rewrite` flag to `evaluate_pipeline.py` with choices `keywords` (original), `sentence` (B), `dual` (C). New helper functions: `_rewrite_sentence()`, `_retrieve_with_dedup()`.

**Evaluation results (v2, k=3, 100 questions):**

| Metric             | keywords (Round 2 baseline) | sentence (B) | dual (C) |
|--------------------|----------------------------|-------------|----------|
| Context Precision  | 0.344                      | 0.521       | 0.489    |
| Context Recall     | 0.348                      | 0.583       | 0.552    |
| Faithfulness       | 0.802                      | 0.701       | 0.722    |
| Answer Relevancy   | 0.645                      | 0.670       | 0.739    |

**Key findings:**
- **Context Precision improved massively**: sentence (0.521) vs keywords (0.344) — +51% improvement! The retrieval stage finally finds more relevant chunks.
- **Context Recall also jumped**: sentence (0.583) vs keywords (0.348) — +68% improvement! The rewritten formal sentences match regulatory text much better.
- **Faithfulness dropped slightly**: sentence (0.701) vs keywords (0.802) — -13%. This is unexpected and may be due to evaluation variance or the LLM producing longer/more detailed answers that are harder to score perfectly.
- **Answer Relevancy improved**: dual (0.739) vs keywords (0.645) — +15%. Dual-path gives the best answer relevancy.
- **Sentence vs Dual trade-off**: Sentence is better at retrieval (Precision/Recall), but Dual gives better Answer Relevancy. The dual approach's deduplication may be introducing slightly less precise context.
- **The keyword→sentence rewrite change confirmed the original diagnosis**: the keyword approach was fundamentally wrong for vector search.

**Decision: `sentence` is the best overall strategy** — it provides the biggest improvement where the system was weakest (retrieval quality), which was identified as the bottleneck in Round 2.
- Best config so far: **v2 + k=3 + sentence rewrite**
- Faithfulness regression needs investigation — may improve with prompt tuning or be within evaluation noise.
- Next steps: investigate Faithfulness drop (possible prompt adjustment), consider adding a Reranker to further improve Context Precision, or try hybrid search (vector + BM25 keyword).

### Round 4: System Prompt tuning for Faithfulness recovery (2026-03-24, in progress)
**Goal:** Recover the Faithfulness score that dropped from 0.802 (Round 2 keywords) to 0.701 (Round 3 sentence) by tightening the generation prompt — without losing the retrieval gains from Round 3.

**Attempt 4a: Ultra-strict prompt** — Added hard constraints:
- "只能使用參考資料回答，嚴禁添加任何常識、推論"
- "如果參考資料中沒有明確答案，直接回答「規章未說明」"
- "用最精練、直接的語氣作答，嚴禁客套話"

| Metric             | Round 3 sentence (old prompt) | 4a strict prompt |
|--------------------|------------------------------|------------------|
| Context Precision  | 0.521                        | 0.514            |
| Context Recall     | 0.583                        | 0.569            |
| Faithfulness       | 0.701                        | **0.656** (worse) |
| Answer Relevancy   | 0.670                        | **0.628** (worse) |
| Faith Pass(>=0.8)  | —                            | 60/100           |

**Root cause:** Llama3 的中文理解力不足以準確判斷「參考資料是否包含答案」。加上嚴格限制後，它矯枉過正，大量把有答案的題目也回「規章未說明」（如：全勤獎金、特休假、資安通報專線等明確有寫在參考資料中的答案）。Gemini 因此給予 Faithfulness=0。

**Attempt 4b: Balanced prompt (evaluated 2026-03-25)** — 調整策略：
- 從「嚴禁回答」改為「鼓勵從資料中提取」：「仔細閱讀參考資料，從中找出與問題相關的條文，直接引用作答」
- 降低「規章未說明」門檻：「只有當參考資料中『完全沒有』任何相關內容時，才回答規章未說明」
- 保留防幻覺機制：「嚴禁添加參考資料中沒有提到的內容」
- 保留簡潔要求：「回答必須簡潔扼要，不要加客套話」

| Metric             | Round 3 sentence (baseline) | 4a strict | **4b balanced** |
|--------------------|----------------------------|-----------|-----------------|
| Context Precision  | 0.521                      | 0.514     | 0.505           |
| Context Recall     | 0.583                      | 0.569     | 0.538           |
| Faithfulness       | 0.701                      | 0.656     | 0.671           |
| Answer Relevancy   | 0.670                      | 0.628     | 0.646           |
| Faith Pass(>=0.8)  | —                          | 60/100    | 61/100          |

**Result:** 4b 未達預期。Faithfulness 僅從 4a 的 0.656 微升至 0.671，仍低於 Round 3 baseline 的 0.701。Context Precision/Recall 也略為下滑。

**Conclusion:** 單靠 prompt tuning 對 Llama3 8B 的 Faithfulness 改善效果有限。Round 3 的 prompt（sentence rewrite, 無額外嚴格限制）仍是目前最佳組合。

**Decision: 回退至 Round 3 prompt 作為當前最佳配置。**
- Best config: **v2 + k=3 + sentence rewrite + Round 3 prompt**
- Best scores: CP=0.521, CR=0.583, Faith=0.701, AR=0.670

**Insight for Llama3 prompt engineering:**
- Llama3 (8B) 對中文指令的遵循度不夠精確，太嚴格的限制會導致「寧可不答也不敢答」
- 適合 Llama3 的 prompt 風格：正面引導（「找出相關條文並引用」）優於負面禁止（「嚴禁回答」）
- 「規章未說明」的觸發條件必須寬鬆（「完全沒有」），否則會誤殺大量正確答案
- 即使改為正面引導（4b），Faithfulness 改善幅度仍有限 — prompt tuning 的天花板已到

**Potential next directions:**
1. ~~加入 Reranker（如 bge-reranker）— 改善 Context Precision~~ ✅ Done (Round 5)
2. ~~Hybrid search（vector + BM25）— 同時改善 Precision 和 Recall~~ ✅ Done (Round 6)
3. 換更大的模型（如 Llama3 70B 或 Qwen2.5-7B）— 根本提升中文理解力與指令遵循度
4. HyDE（Hypothetical Document Embedding）— 讓 LLM 先假裝回答再用假答案做 vector search
5. 多輪檢索（Iterative Retrieval）— 第一輪不夠再改寫 query 搜第二次

### Round 5: Reranker — bge-reranker-v2-m3 (2026-03-25)
**改動**：在 #9 最佳配置基礎上加入二階段檢索：sentence rewrite → 取 top 9 → bge-reranker-v2-m3 重排 → top 3 → LLM
**實作**：`evaluate_pipeline.py` 新增 `_rerank()` 函數 + `reranker` 參數；`run_reranker_test.py` 為獨立測試腳本

| Metric | #9 (無 reranker) | #11 (+ reranker) | 變化 |
|--------|-----------------|-----------------|------|
| CP | 0.550 | 0.573 | +4.2% |
| CR | 0.485 | 0.506 | +4.3% |
| Faith | 0.685 | 0.683 | -0.3% |
| AR | 0.524 | 0.537 | +2.5% |

**結論**：Reranker 帶來穩定但溫和的改善。v2 只有 76 個 chunks，候選池較小限制了 reranker 的發揮空間。
**最佳配置**：v2 + k=3 + sentence rewrite + bge-reranker-v2-m3
**硬體成本**：GPU 記憶體增加 ~1.6GB (8860→10434MB)，推論速度幾乎不變 (3.3→3.6s/題)

### Round 6: Hybrid Search — BM25 + Vector + Reranker (2026-03-26)
**改動**：在 Round 5 最佳配置基礎上加入 BM25 關鍵字檢索，與 Vector 語義檢索並行，透過 Reciprocal Rank Fusion (RRF) 合併結果後再由 Reranker 重排。
**動機**：Vector search 對中文法條編號（如「第十二條」）和專有名詞（如「全勤獎金」「特休假」）的精確匹配能力有限，BM25 可補強這些場景。

**檢索流程**：
```
sentence rewrite → Vector search (top 9) + BM25 search (top 9)
                 → RRF 合併 (rrf_k=60)
                 → bge-reranker-v2-m3 重排 → top 3 → LLM
```

**實作**：
- 套件：`rank_bm25`（BM25Okapi）+ `jieba`（中文分詞）
- `evaluate_pipeline.py` 新增：`_tokenize_chinese()`、`build_bm25_index()`、`_bm25_search()`、`_hybrid_retrieve()`
- `rag_query()` 和 `generate_answers()` 新增 `bm25_index`/`bm25_docs` 參數
- `run_hybrid_test.py` 為獨立測試腳本

| Metric | #11 R5 (無 BM25) | #12 R6 (+ BM25 Hybrid) | 變化 |
|--------|-----------------|----------------------|------|
| CP | 0.573 | **0.650** | **+13.4%** |
| CR | 0.506 | **0.601** | **+18.8%** |
| Faith | 0.683 | **0.743** | **+8.8%** |
| AR | 0.537 | **0.629** | **+17.1%** |

**關鍵發現**：
- **四項指標全面提升**，且幅度遠大於 Round 5 Reranker 的溫和改善
- CP/CR 提升最顯著（+13-19%）：BM25 對中文精確匹配（法條編號、專有名詞）的補強效果明確
- Faithfulness 突破 0.74：更精準的 context 讓 LLM 回答更忠實
- 推論速度幾乎不變（3.6→3.8s/題）：BM25 跑在 CPU 上，不占 GPU 資源
- GPU 記憶體無明顯變化（~10,300MB）

**最佳配置**：v2 + k=3 + sentence rewrite + BM25 hybrid (RRF) + bge-reranker-v2-m3
**硬體成本**：與 Round 5 相比，CPU 略增但 GPU 無變化，性價比極高

### Round 7: HyDE — Hypothetical Document Embedding (2026-03-26)
**改動**：在 Round 6 最佳配置基礎上加入 HyDE，讓 LLM 先產生「假設性的規章回答」，用這段假答案做 Vector search（語義更接近法規文件），BM25 仍用 sentence rewrite 查詢。
**動機**：Vector search 的語義匹配容易因用戶口語與法規書面語的語義落差而撈錯 chunk。HyDE 的假答案語言風格更接近法規文件，可縮小 embedding 空間的距離。

**檢索流程**：
```
1. sentence rewrite → 正式書面語查詢（供 BM25）
2. HyDE → LLM 產生假設性規章回答（供 Vector search）
3. Vector search(假答案, top 9) + BM25(書面語, top 9)
   → RRF 合併 → bge-reranker-v2-m3 重排 → top 3 → LLM 正式回答
```

**實作**：
- `evaluate_pipeline.py` 新增 `_hyde_generate()` 函數
- `_hybrid_retrieve()` 新增 `vector_query` 參數（允許 vector/BM25 用不同查詢）
- `rag_query()` + `generate_answers()` 新增 `use_hyde` 參數
- `run_hyde_test.py` 為獨立測試腳本

| Metric | #12 R6 (無 HyDE) | #13 R7 (+ HyDE) | 變化 |
|--------|-----------------|-----------------|------|
| CP | 0.650 | **0.661** | +1.7% |
| CR | 0.601 | **0.615** | +2.3% |
| Faith | 0.743 | **0.772** | **+3.9%** |
| AR | 0.629 | **0.665** | **+5.7%** |

**關鍵發現**：
- **四項指標均有提升**，但幅度較 Round 6 (BM25) 溫和
- **AR 提升最明顯 (+5.7%)**：HyDE 假答案幫助 vector search 撈到更相關的 context，讓 LLM 回答更貼切
- **Faithfulness 提升 +3.9%**：更精準的 context 減少幻覺
- **CP/CR 提升有限 (+1.7%/+2.3%)**：HyDE 對 Llama3 8B 來說，假答案品質受中文能力限制
- **推論速度增加一倍**（3.8→7.4s/題）：每題多一次 LLM 呼叫用於產生假答案
- **具體改善案例**：
  - Q67（薪水自動調漲）：R6 幻覺回答「會」→ R7 正確回答「不會」
  - Q18（颱風停班）：R6 答「需要 WFH」→ R7 答「不需要」
  - Q80（公發測試機帶回家）：R6 答「規章未說明」→ R7 正確引用設備離場登記
  - Q84（宗教彈性工時）：R6 幻覺答「可以」→ R7 正確答「不允許」

**最佳配置**：v2 + k=3 + sentence rewrite + HyDE + BM25 hybrid (RRF) + bge-reranker-v2-m3
**硬體成本**：GPU 記憶體持平（~10,300MB），推論時間增加一倍（7.4s/題 vs 3.8s/題）

**取捨考量**：HyDE 帶來穩定但溫和的提升，代價是推論速度減半。在離線評測場景可接受，但如果是即時回答場景可能需要權衡。

### Round 8: Chunk 內容增強 — 標題前綴 (2026-03-26)
**改動**：在 chunk 的 `page_content` 前面加上結構化標題前綴（如 `【員工請假管理辦法 > 假別規定 > 第3條 事假】`），讓 embedding 和 BM25 都能匹配到標題中的關鍵字。原始 `.md` 檔案不動，只在 embedding 時拼接。

**兩種 Splitter 對比**：
- **v3 (MarkdownHeaderTextSplitter + 標題前綴)**：沿用既有 splitter，metadata 有三級（document_title, chapter, article），前綴最完整
- **v4 (LlamaIndex MarkdownNodeParser + 標題前綴)**：不同的切分引擎，text 本身包含 header，header_path 提供層級路徑，前綴只有兩級

**實作**：
- `embed_docs.py` 新增 `--version v3/v4`、`_build_title_prefix()`、`load_and_split_markdown_v3()`、`load_and_split_markdown_v4()`
- `evaluate_pipeline.py` 新增 `DB_DIR_V3`/`DB_DIR_V4` 支援
- `run_chunk_enhanced_test.py` 為獨立測試腳本（支援 `--db v3/v4`）
- 套件新增：`llama-index-core`（v4 所需）

| Metric | R7 v2 (無前綴) | R8a v3 (Header+前綴) | R8b v4 (Node+前綴) |
|--------|--------------|---------------------|-------------------|
| CP | 0.661 | **0.668** (+1.1%) | 0.664 (+0.5%) |
| CR | 0.615 | **0.622** (+1.1%) | 0.618 (+0.5%) |
| Faith | 0.772 | **0.778** (+0.8%) | 0.769 (-0.4%) |
| AR | 0.665 | **0.673** (+1.2%) | 0.659 (-0.9%) |

**關鍵發現**：
- **v3 (HeaderSplitter) 優於 v4 (NodeParser)**：v3 四項指標全部最高
- **v3 對比 R7 提升溫和**（各項 +1%）：因為 v2 的 `strip_headers=False` 已保留標題在 content 中，加前綴的增量效果有限
- **v4 略遜於 v3**：NodeParser 的前綴只有兩級（缺 article 級），資訊量少了一級
- **具體改善案例**：Q74（挖比特幣）— 標題前綴讓 BM25 命中了「第 5 條 終端設備防護」中的「加密貨幣挖礦程式」禁令

**決策**：v3 為 Round 8 最佳配置，但提升幅度有限。標題前綴的效果在 v2 `strip_headers=False` 的基礎上邊際遞減。
**最佳配置**：v3 + k=3 + sentence rewrite + HyDE + BM25 hybrid (RRF) + bge-reranker-v2-m3

### 全歷程數據總表（含硬體效能）

| # | Round | 改動重點 | CP | CR | Faith | AR | s/題 | GPU Peak (MB) |
|---|-------|---------|------|------|-------|------|------|-------------|
| 1 | Original | mxbai-embed + v1 + keywords | 0.193 | 0.159 | 0.517 | 0.264 | 4.0 | 9,731 |
| 2 | Baseline | **bge-m3 (GPU)** | 0.411 | 0.357 | 0.627 | 0.411 | 3.7 | 9,733 |
| 3 | R1 | k=5 | 0.389 | 0.335 | 0.563 | 0.357 | 4.4 | 8,860 |
| 4 | R1 | k=7 | 0.383 | 0.328 | 0.541 | 0.350 | 4.7 | 8,860 |
| 5 | R1 | k=10 | 0.395 | 0.336 | 0.510 | 0.351 | 5.0 | 8,860 |
| 6 | R2 | **v2 MarkdownHeader** + k=3 | 0.380 | 0.341 | 0.600 | 0.394 | 3.3 | 8,860 |
| 7 | R2 | v2 + k=5 | 0.373 | 0.327 | 0.559 | 0.375 | 3.7 | 8,860 |
| 8 | R2 | v2 + k=7 | 0.365 | 0.321 | 0.574 | 0.363 | 3.8 | 8,860 |
| 9 | R3 | **sentence rewrite** | 0.550 | 0.485 | 0.685 | 0.524 | 3.3 | 8,860 |
| 10 | R3 | dual rewrite | 0.547 | 0.482 | 0.679 | 0.520 | 3.4 | 8,860 |
| 11 | R5 | **+ Reranker** | 0.573 | 0.506 | 0.683 | 0.537 | 3.6 | 10,434 |
| 12 | R6 | **+ BM25 Hybrid** | 0.650 | 0.601 | 0.743 | 0.629 | 3.8 | 10,324 |
| 13 | R7 | **+ HyDE** | 0.661 | 0.615 | 0.772 | 0.665 | 7.4 | 10,326 |
| 14 | **R8a** | **v3 標題前綴 (HeaderSplitter)** | **0.668** | **0.622** | **0.778** | **0.673** | 8.3 | 12,436 |
| 15 | R8b | v4 標題前綴 (NodeParser) | 0.664 | 0.618 | 0.769 | 0.659 | 8.1 | 12,526 |

> 硬體環境：GCP VM + NVIDIA L4 GPU (24GB VRAM) | LLM: Llama3 8B (Ollama) | Embedding: BAAI/bge-m3 (CUDA)

### 從原始模型到最終配置的總提升
| Metric | #1 Original | #14 Best (R8a v3) | 提升 |
|--------|------------|------------------|------|
| CP | 0.193 | 0.668 | **+246%** |
| CR | 0.159 | 0.622 | **+291%** |
| Faith | 0.517 | 0.778 | **+50%** |
| AR | 0.264 | 0.673 | **+155%** |

### 小結：8 輪迭代的關鍵洞察

**1. 投資報酬率最高的三項改動（佔總提升 ~85%）**
- **Embedding 切換**（Original→Baseline）：bge-m3 取代 mxbai-embed，四項指標翻倍。成本幾乎為零。
- **Query Rewrite 策略**（R2→R3）：keywords→sentence，CP/CR 提升 +45%/+42%。成本為零。
- **BM25 Hybrid Search**（R5→R6）：四項指標全面提升 +13~19%，CPU 運算不增加 GPU 負擔。

**2. 穩定但溫和的改動**
- Reranker（R3→R5）：+4% CP/CR，但增加 ~1.6GB GPU 記憶體
- HyDE（R6→R7）：+4~6% Faith/AR，但推論速度減半（3.8→7.4s/題）
- Chunk 標題前綴（R7→R8a）：+1% 全面提升，邊際遞減

**3. 失敗或無效的嘗試**
- k 值調高（R1 k=5/7/10）：Recall 微升但 Faithfulness 下降，噪音太多
- 嚴格 prompt（R4a/4b）：Llama3 8B 中文理解力不足，矯枉過正
- MarkdownNodeParser（R8b v4）：兩級前綴不如三級，效果略遜 v3

**4. 硬體成本控制**
- GPU 記憶體從 9.7GB 增至 12.4GB（+28%），仍在 L4 的 24GB 內有充裕空間
- 推論速度從 3.3s/題 增至 8.3s/題（+152%），主要來自 HyDE 的額外 LLM 呼叫
- 如果延遲敏感，可關閉 HyDE 回到 3.8s/題（R6 配置），僅損失 ~2% 各項指標

**5. 剩餘瓶頸（不換模型難以突破）**
- ~10 題持續檢索失敗（喪假、飛機票、機房滅火、DB 匿名化等）— 規章中缺乏對應關鍵字的 chunk
- Llama3 8B 中文閱讀理解限制 — 偶爾從正確 context 中讀出錯誤答案（如 25 萬簽核選錯級別）
- 「規章未說明」判斷力不足 — 部分幻覺回答仍無法根治

## 面試簡報

### 第一版 (0326)
- **用途**：個人面試求職，向主管展現 LLM 本地端部署與迭代優化能力
- **檔案**：`07_evaluation_results/LLM_Local_Deploy_Presentation_v1_0326.pptx`
- **產生腳本**：`generate_ppt.py`（python-pptx）
- **風格**：深色背景（#0F172A），專業技術簡報
- **架構（10 頁）**：
  1. 封面（LLM 本地端部署）
  2. 專案簡述（目標 + 方法論）
  3. RAG 四大評估指標說明（含意義、舉例、業界標準，引用 RAGAS 論文）
  4. 原生模型表現 + 2 個失敗範例（密碼→搜到績效考核、代打卡→搜到教育訓練）
  5. 改良①：Embedding 切換 + 結構化切分（CP +113%）+ 失敗範例（keyword 拆散語義）
  6. 改良②：Query Rewrite keywords→sentence（CP +45%）+ 失敗範例（高鐵 vs 飛機混淆）
  7. 改良③：Reranker + BM25 Hybrid（四項 +13~24%）+ 失敗範例（薪水幻覺）
  8. 改良④：HyDE（AR +5.7%）+ 改善案例（薪水幻覺修正）
  9. 改良⑤：Chunk 標題前綴 v3 vs v4 + 改善案例（挖比特幣精確命中）
  10. 結論：總提升 CP+246% CR+291% Faith+50% AR+155% + 後續改良方向
- **每頁改良包含**：改動說明、數據對比表、失敗/改善範例（註明問題診斷+下一步方向）、硬體效能數據

## 全面重新測評（2026-03-25 進行中）

### 狀態：✅ 全部完成（2026-03-25）
以 Claude Code 取代 Gemini API 作為評測模型，從原始模型重新跑完所有階段（含 Round 5 Reranker）。
最新 commit: `90307c0` 已 push 至 GitHub。

### 推論進度（run_all_tests.py，本地執行）— ✅ 全部完成（2026-03-25, 66.4 min）
| # | run_label | answers.json | hardware.json |
|---|-----------|-------------|---------------|
| 1 | `original_v1-mxbai_k3_rw-keywords` | ✅ `_20260325_114498` | ✅ |
| 2 | `baseline_v1-bge_k3_rw-keywords` | ✅ `_20260325_115536` | ✅ |
| 3 | `R1_dbv1_k5_rw-keywords` | ✅ `_20260325_120227` | ✅ |
| 4 | `R1_dbv1_k7_rw-keywords` | ✅ `_20260325_121009` | ✅ |
| 5 | `R1_dbv1_k10_rw-keywords` | ✅ `_20260325_121832` | ✅ |
| 6 | `R2_dbv2_k3_rw-keywords` | ✅ `_20260325_122651` | ✅ |
| 7 | `R2_dbv2_k5_rw-keywords` | ✅ `_20260325_123339` | ✅ |
| 8 | `R2_dbv2_k7_rw-keywords` | ✅ `_20260325_123949` | ✅ |
| 9 | `R3_dbv2_k3_rw-sentence` | ✅ `_20260325_124617` | ✅ |
| 10 | `R3_dbv2_k3_rw-dual` | ✅ `_20260325_125148` | ✅ |

### 評測進度（Claude Code 讀取 answers.json 評分）
| # | run_label | 狀態 | 結果 |
|---|-----------|------|------|
| 1 | `original_v1-mxbai_k3_rw-keywords` | ✅ 完成 | CP=0.193, CR=0.159, Faith=0.517, AR=0.264 |
| 2 | `baseline_v1-bge_k3_rw-keywords` | ✅ 完成 | CP=0.411, CR=0.357, Faith=0.627, AR=0.411 |
| 3 | `R1_dbv1_k5_rw-keywords` | ✅ 完成 | CP=0.389, CR=0.335, Faith=0.563, AR=0.357 |
| 4 | `R1_dbv1_k7_rw-keywords` | ✅ 完成 | CP=0.383, CR=0.328, Faith=0.541, AR=0.350 |
| 5 | `R1_dbv1_k10_rw-keywords` | ✅ 完成 | CP=0.395, CR=0.336, Faith=0.510, AR=0.351 |
| 6 | `R2_dbv2_k3_rw-keywords` | ✅ 完成 | CP=0.380, CR=0.341, Faith=0.600, AR=0.394 |
| 7 | `R2_dbv2_k5_rw-keywords` | ✅ 完成 | CP=0.373, CR=0.327, Faith=0.559, AR=0.375 |
| 8 | `R2_dbv2_k7_rw-keywords` | ✅ 完成 | CP=0.365, CR=0.321, Faith=0.574, AR=0.363 |
| 9 | `R3_dbv2_k3_rw-sentence` | ✅ 完成 | CP=0.550, CR=0.485, Faith=0.685, AR=0.524 |
| 10 | `R3_dbv2_k3_rw-dual` | ✅ 完成 | CP=0.547, CR=0.482, Faith=0.679, AR=0.520 |

| 11 | `R5_dbv2_k3_rw-sentence_reranker` | ✅ 完成 | CP=0.573, CR=0.506, Faith=0.683, AR=0.537 |
| 12 | `R6_dbv2_k3_rw-sentence_hybrid-bm25_reranker` | ✅ 完成 | CP=0.650, CR=0.601, Faith=0.743, AR=0.629 |
| 13 | `R7_dbv2_k3_rw-sentence_hyde_hybrid-bm25_reranker` | ✅ 完成 | CP=0.661, CR=0.615, Faith=0.772, AR=0.665 |
| 14 | `R8a_dbv3_k3_rw-sentence_hyde_hybrid-bm25_reranker` | ✅ 完成 | CP=0.668, CR=0.622, Faith=0.778, AR=0.673 |
| 15 | `R8b_dbv4_k3_rw-sentence_hyde_hybrid-bm25_reranker` | ✅ 完成 | CP=0.664, CR=0.618, Faith=0.769, AR=0.659 |

> 評測方式：Claude Code 直接讀取 answers.json + 規章文件評分（不呼叫 API）
> 輔助腳本：`06_core_app/eval_helper.py`（read 讀批次 / write 寫評分 / report 產圖表）
> 評分 CSV 位於 `07_evaluation_results/scores/{run_label}.csv`，每完成一份即更新此表
> 每批完成後更新此表和 CHANGELOG.txt，確保可從任意中斷點續接。
