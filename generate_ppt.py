#!/usr/bin/env python3
"""
面試用簡報產生器 — LLM 本地端部署專案
第一版 (0326)
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ============================================================
# 色彩主題（深色背景）
# ============================================================
BG_DARK = RGBColor(0x0F, 0x17, 0x2A)       # 主背景
CARD_BG = RGBColor(0x1E, 0x29, 0x3B)        # 卡片背景
TEXT_WHITE = RGBColor(0xF8, 0xFA, 0xFC)      # 主文字
TEXT_SUB = RGBColor(0x94, 0xA3, 0xB8)        # 次要文字
ACCENT_BLUE = RGBColor(0x3B, 0x82, 0xF6)    # 強調藍
ACCENT_GREEN = RGBColor(0x10, 0xB9, 0x81)   # 成功綠
ACCENT_RED = RGBColor(0xEF, 0x44, 0x44)     # 警告紅
ACCENT_YELLOW = RGBColor(0xF5, 0x9E, 0x0B)  # 黃色
ACCENT_PURPLE = RGBColor(0xA7, 0x8B, 0xFA)  # 紫色
BORDER_COLOR = RGBColor(0x33, 0x41, 0x55)    # 邊框

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)


def set_bg(slide, color=BG_DARK):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height, text, font_size=14,
                color=TEXT_WHITE, bold=False, align=PP_ALIGN.LEFT, font_name="Microsoft JhengHei"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = align
    return txBox


def add_card(slide, left, top, width, height, fill_color=CARD_BG):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    shape.shadow.inherit = False
    # 調整圓角
    shape.adjustments[0] = 0.05
    return shape


def add_multiline(slide, left, top, width, height, lines, font_size=13, line_spacing=1.3):
    """lines = [(text, color, bold), ...]"""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, (text, color, bold) in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.bold = bold
        p.font.name = "Microsoft JhengHei"
        p.space_after = Pt(font_size * (line_spacing - 1))
    return txBox


def add_metric_row(slide, left, top, width, label, before, after, improvement):
    """添加一行指標對比"""
    add_textbox(slide, left, top, 2.5, 0.35, label, 13, TEXT_SUB)
    add_textbox(slide, left + 2.5, top, 1.5, 0.35, f"{before:.3f}", 13, TEXT_SUB)
    add_textbox(slide, left + 4.0, top, 1.5, 0.35, f"{after:.3f}", 13, ACCENT_GREEN, bold=True)
    color = ACCENT_GREEN if improvement > 0 else ACCENT_RED
    sign = "+" if improvement > 0 else ""
    add_textbox(slide, left + 5.5, top, 1.5, 0.35, f"{sign}{improvement:.1f}%", 13, color, bold=True)


# ============================================================
# P1: 封面
# ============================================================
slide1 = prs.slides.add_slide(prs.slide_layouts[6])  # blank
set_bg(slide1)

# 裝飾線
line_shape = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1), Inches(2.8), Inches(1.5), Inches(0.06))
line_shape.fill.solid()
line_shape.fill.fore_color.rgb = ACCENT_BLUE
line_shape.line.fill.background()

add_textbox(slide1, 1, 3.0, 11, 1.0, "LLM 本地端部署", 44, TEXT_WHITE, bold=True)
add_textbox(slide1, 1, 4.0, 11, 0.8, "全離線 RAG 系統的迭代優化與評測", 22, TEXT_SUB)
add_textbox(slide1, 1, 5.2, 11, 0.5, "Llama3 8B  |  BAAI/bge-m3  |  ChromaDB  |  GCP L4 GPU", 16, ACCENT_BLUE)
add_textbox(slide1, 1, 6.5, 11, 0.4, "2026.03", 14, TEXT_SUB)


# ============================================================
# P2: 專案簡述
# ============================================================
slide2 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide2)

add_textbox(slide2, 0.8, 0.4, 5, 0.6, "專案簡述", 28, TEXT_WHITE, bold=True)

# 左側：目標
add_card(slide2, 0.8, 1.3, 5.5, 5.5)
add_textbox(slide2, 1.1, 1.5, 5.0, 0.5, "目標", 18, ACCENT_BLUE, bold=True)
add_multiline(slide2, 1.1, 2.1, 5.0, 4.5, [
    ("在 GCP VM (L4 GPU) 上部署全離線 RAG 系統", TEXT_WHITE, False),
    ("", TEXT_WHITE, False),
    ("知識庫：11 份公司內部規章（繁體中文）", TEXT_SUB, False),
    ("模型：Llama3 8B（透過 Ollama 本地推論）", TEXT_SUB, False),
    ("Embedding：BAAI/bge-m3（CUDA 加速）", TEXT_SUB, False),
    ("向量庫：ChromaDB", TEXT_SUB, False),
    ("後端：FastAPI + 單頁式前端", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("安全限制：所有推論必須離線完成", ACCENT_YELLOW, False),
    ("僅評測階段可對外呼叫 API", ACCENT_YELLOW, False),
], font_size=14)

# 右側：方法論
add_card(slide2, 6.8, 1.3, 5.7, 5.5)
add_textbox(slide2, 7.1, 1.5, 5.2, 0.5, "方法論", 18, ACCENT_BLUE, bold=True)
add_multiline(slide2, 7.1, 2.1, 5.2, 4.5, [
    ("以 100 題 Golden Dataset 進行系統性評測", TEXT_WHITE, False),
    ("", TEXT_WHITE, False),
    ("每一輪改動遵循相同流程：", TEXT_SUB, False),
    ("  1. 分析瓶頸 → 提出假設", TEXT_SUB, False),
    ("  2. 實作改動", TEXT_SUB, False),
    ("  3. 跑完 100 題推論", TEXT_SUB, False),
    ("  4. 用 Claude Sonnet 嚴格閱卷", TEXT_SUB, False),
    ("  5. 對比數據決策", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("共完成 8 輪迭代、15 組測試", ACCENT_GREEN, True),
], font_size=14)


# ============================================================
# P3: 四大評估指標
# ============================================================
slide3 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide3)

add_textbox(slide3, 0.8, 0.4, 10, 0.6, "RAG 四大評估指標", 28, TEXT_WHITE, bold=True)
add_textbox(slide3, 0.8, 0.95, 10, 0.4, "Based on RAGAS: Automated Evaluation of Retrieval Augmented Generation", 12, TEXT_SUB)

metrics = [
    ("Context Precision", "檢索精準度",
     "檢索到的文件是否與問題相關？",
     "問「請假規定」→ 撈到「請假辦法」✓\n問「請假規定」→ 撈到「採購程序」✗",
     "≥ 0.7", ACCENT_BLUE),
    ("Context Recall", "檢索召回率",
     "答案所需的資訊是否都被檢索到？",
     "問「婚假幾天？多久請完？」\n→ 撈到天數但漏掉期限 = 低 Recall",
     "≥ 0.7", ACCENT_YELLOW),
    ("Faithfulness", "忠實度",
     "模型回答是否忠於檢索到的參考資料？\n（有沒有幻覺 Hallucination）",
     "參考資料寫「8 日婚假」\n→ 模型回答「10 日」= 幻覺 ✗",
     "≥ 0.85", ACCENT_GREEN),
    ("Answer Relevancy", "回答相關度",
     "回答是否直接、完整地回應了問題？",
     "問「密碼規定」→ 回答密碼規定 ✓\n問「密碼規定」→ 回答打卡規定 ✗",
     "≥ 0.7", ACCENT_PURPLE),
]

for i, (name_en, name_zh, meaning, example, ideal, color) in enumerate(metrics):
    col = i % 2
    row = i // 2
    x = 0.8 + col * 6.2
    y = 1.5 + row * 2.9

    add_card(slide3, x, y, 5.8, 2.6)
    add_textbox(slide3, x + 0.2, y + 0.15, 3.5, 0.4, f"{name_en}", 16, color, bold=True)
    add_textbox(slide3, x + 0.2, y + 0.55, 3.5, 0.3, name_zh, 12, TEXT_SUB)

    # 意義
    add_textbox(slide3, x + 0.2, y + 0.9, 5.3, 0.6, meaning, 11, TEXT_WHITE)

    # 範例
    add_textbox(slide3, x + 0.2, y + 1.55, 5.3, 0.7, example, 10, TEXT_SUB)

    # 業界標準
    add_textbox(slide3, x + 3.8, y + 0.15, 1.8, 0.35, f"業界標準 {ideal}", 11, ACCENT_GREEN)


# ============================================================
# P4: 原生模型表現
# ============================================================
slide4 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide4)

add_textbox(slide4, 0.8, 0.4, 10, 0.6, "原生模型表現 — Llama3 8B + mxbai-embed", 28, TEXT_WHITE, bold=True)

# 數據表格
add_card(slide4, 0.8, 1.2, 5.5, 2.8)
add_textbox(slide4, 1.1, 1.35, 5, 0.4, "原始模型指標", 16, ACCENT_RED, bold=True)

headers = [("指標", 0), ("數值", 2.5), ("業界標準", 4.0)]
for h, offset in headers:
    add_textbox(slide4, 1.1 + offset, 1.8, 1.5, 0.3, h, 12, TEXT_SUB, bold=True)

orig_data = [
    ("Context Precision", 0.193, "≥ 0.70"),
    ("Context Recall", 0.159, "≥ 0.70"),
    ("Faithfulness", 0.517, "≥ 0.85"),
    ("Answer Relevancy", 0.264, "≥ 0.70"),
]
for j, (m, v, ideal) in enumerate(orig_data):
    y = 2.15 + j * 0.38
    add_textbox(slide4, 1.1, y, 2.5, 0.35, m, 13, TEXT_WHITE)
    add_textbox(slide4, 3.6, y, 1.2, 0.35, f"{v:.3f}", 13, ACCENT_RED, bold=True)
    add_textbox(slide4, 5.1, y, 1.0, 0.35, ideal, 12, TEXT_SUB)

add_textbox(slide4, 1.1, 3.65, 5, 0.3, "推論速度 4.0 s/題  |  GPU 9,731 MB", 11, TEXT_SUB)

# 失敗範例 1
add_card(slide4, 6.8, 1.2, 5.7, 2.5)
add_textbox(slide4, 7.1, 1.35, 5.2, 0.35, "失敗範例 ① 檢索完全錯誤", 14, ACCENT_RED, bold=True)
add_multiline(slide4, 7.1, 1.75, 5.2, 2.0, [
    ("Q：密碼設定有什麼複雜度要求？", TEXT_WHITE, True),
    ("標準答案：12 字元、含大小寫+數字+符號三項、90 天更換", ACCENT_GREEN, False),
    ("", TEXT_WHITE, False),
    ("模型回答：根據《員工績效考核辦法》第 4 條 遲到、", ACCENT_RED, False),
    ("早退與緩衝期機制... 無相關內容", ACCENT_RED, False),
    ("", TEXT_WHITE, False),
    ("診斷：mxbai embedding 語義能力差，問密碼搜到績效考核", ACCENT_YELLOW, False),
], font_size=11)

# 失敗範例 2
add_card(slide4, 6.8, 4.0, 5.7, 2.5)
add_textbox(slide4, 7.1, 4.15, 5.2, 0.35, "失敗範例 ② 嚴重語義漂移", 14, ACCENT_RED, bold=True)
add_multiline(slide4, 7.1, 4.55, 5.2, 2.0, [
    ("Q：幫同事代打卡被抓到，最嚴重的懲處？", TEXT_WHITE, True),
    ("標準答案：記大過一次，情節嚴重者予以免職", ACCENT_GREEN, False),
    ("", TEXT_WHITE, False),
    ("模型回答：根據《員工績效考核辦法》... 獎金發放、", ACCENT_RED, False),
    ("薪資調整... （完全不是懲處）", ACCENT_RED, False),
    ("", TEXT_WHITE, False),
    ("診斷：搜到教育訓練而非考勤制度 → 需更換 Embedding 模型", ACCENT_YELLOW, False),
], font_size=11)

# 下一步
add_card(slide4, 0.8, 4.3, 5.5, 2.2)
add_textbox(slide4, 1.1, 4.45, 5, 0.35, "下一步改動方向", 16, ACCENT_BLUE, bold=True)
add_multiline(slide4, 1.1, 4.85, 5, 1.5, [
    ("1. 更換 Embedding 模型：mxbai → BAAI/bge-m3 (GPU)", TEXT_WHITE, False),
    ("   → 多語言語義理解能力更強", TEXT_SUB, False),
    ("2. 改善 Chunking 策略：固定 500 字切分 → 按文件結構切分", TEXT_WHITE, False),
    ("   → 避免跨條文污染", TEXT_SUB, False),
], font_size=12)


# ============================================================
# P5: Embedding 切換 + Chunking 改良（最大跳躍）
# ============================================================
slide5 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide5)

add_textbox(slide5, 0.8, 0.4, 10, 0.6, "改良 ① Embedding 切換 + 結構化切分", 28, TEXT_WHITE, bold=True)
add_textbox(slide5, 0.8, 0.95, 10, 0.4, "Original → Baseline → Round 2", 14, TEXT_SUB)

# 改動說明
add_card(slide5, 0.8, 1.4, 7.2, 2.2)
add_textbox(slide5, 1.1, 1.55, 6.8, 0.35, "兩項關鍵改動", 16, ACCENT_BLUE, bold=True)
add_multiline(slide5, 1.1, 1.95, 6.8, 1.5, [
    ("A. Embedding 模型：OllamaEmbeddings(mxbai) → HuggingFaceEmbeddings(bge-m3, CUDA)", TEXT_WHITE, False),
    ("   bge-m3 對繁體中文的語義理解遠優於 mxbai，且使用 GPU 加速", TEXT_SUB, False),
    ("B. Chunking 策略：RecursiveCharacterTextSplitter(500) → MarkdownHeaderTextSplitter", TEXT_WHITE, False),
    ("   按 # / ## / ### 文件結構切分，每個 chunk 對應一條法規，含 metadata", TEXT_SUB, False),
], font_size=12)

# 指標對比
add_card(slide5, 0.8, 3.9, 7.2, 3.0)
add_textbox(slide5, 1.1, 4.05, 6.8, 0.35, "指標對比", 16, ACCENT_GREEN, bold=True)

add_textbox(slide5, 1.1, 4.45, 2.5, 0.3, "指標", 12, TEXT_SUB, bold=True)
add_textbox(slide5, 3.6, 4.45, 1.5, 0.3, "Original", 12, TEXT_SUB, bold=True)
add_textbox(slide5, 5.1, 4.45, 1.5, 0.3, "Baseline", 12, TEXT_SUB, bold=True)
add_textbox(slide5, 6.3, 4.45, 1.5, 0.3, "提升", 12, TEXT_SUB, bold=True)

data5 = [
    ("Context Precision", 0.193, 0.411, 113.0),
    ("Context Recall", 0.159, 0.357, 124.5),
    ("Faithfulness", 0.517, 0.627, 21.3),
    ("Answer Relevancy", 0.264, 0.411, 55.7),
]
for j, (m, before, after, imp) in enumerate(data5):
    y = 4.8 + j * 0.38
    add_textbox(slide5, 1.1, y, 2.5, 0.35, m, 13, TEXT_WHITE)
    add_textbox(slide5, 3.6, y, 1.2, 0.35, f"{before:.3f}", 13, TEXT_SUB)
    add_textbox(slide5, 5.1, y, 1.2, 0.35, f"{after:.3f}", 13, ACCENT_GREEN, bold=True)
    add_textbox(slide5, 6.3, y, 1.2, 0.35, f"+{imp:.0f}%", 13, ACCENT_GREEN, bold=True)

add_textbox(slide5, 1.1, 6.5, 6, 0.3, "推論 3.7 s/題  |  GPU 9,733 MB（記憶體不變，速度更快）", 11, TEXT_SUB)

# 失敗範例
add_card(slide5, 8.3, 1.4, 4.3, 5.5)
add_textbox(slide5, 8.5, 1.55, 4.0, 0.35, "仍存在的問題", 14, ACCENT_YELLOW, bold=True)
add_multiline(slide5, 8.5, 1.95, 4.0, 5.0, [
    ("Q：特休假沒休完會怎麼處理？", TEXT_WHITE, True),
    ("標準答案：全數折發工資", ACCENT_GREEN, False),
    ("", TEXT_WHITE, False),
    ("Keyword Rewrite 將問題拆成：", TEXT_SUB, False),
    ("「特休」「沒休」「處理」", ACCENT_RED, False),
    ("", TEXT_WHITE, False),
    ("這些零散關鍵字破壞了語義完整性，", TEXT_WHITE, False),
    ("向量搜索反而搜到 Git Commit 規範", ACCENT_RED, False),
    ("", TEXT_WHITE, False),
    ("診斷：Keyword Rewrite 拆散語義，", ACCENT_YELLOW, False),
    ("不適合 Vector Search", ACCENT_YELLOW, False),
    ("", TEXT_WHITE, False),
    ("→ 下一步：改用 Sentence Rewrite", ACCENT_BLUE, True),
    ("   保留完整句子語義", ACCENT_BLUE, False),
], font_size=11)


# ============================================================
# P6: Query Rewrite 策略優化
# ============================================================
slide6 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide6)

add_textbox(slide6, 0.8, 0.4, 10, 0.6, "改良 ② Query Rewrite 策略優化", 28, TEXT_WHITE, bold=True)
add_textbox(slide6, 0.8, 0.95, 10, 0.4, "Round 2 → Round 3  |  keywords → sentence rewrite", 14, TEXT_SUB)

# 改動說明
add_card(slide6, 0.8, 1.4, 7.2, 2.0)
add_textbox(slide6, 1.1, 1.55, 6.8, 0.35, "核心改動", 16, ACCENT_BLUE, bold=True)
add_multiline(slide6, 1.1, 1.95, 6.8, 1.3, [
    ("Before：把口語問題拆成 3~5 個離散關鍵字 → 破壞語義向量品質", ACCENT_RED, False),
    ("After ：把口語問題改寫為一句完整的正式書面語 → 保留語義完整性", ACCENT_GREEN, False),
    ("", TEXT_WHITE, False),
    ("例：「加班可以換補休嗎？」→「員工加班時數是否可轉換為補休假？」", TEXT_SUB, False),
], font_size=12)

# 指標對比
add_card(slide6, 0.8, 3.7, 7.2, 3.0)
add_textbox(slide6, 1.1, 3.85, 6.8, 0.35, "指標對比（本次提升最大）", 16, ACCENT_GREEN, bold=True)

add_textbox(slide6, 1.1, 4.25, 2.5, 0.3, "指標", 12, TEXT_SUB, bold=True)
add_textbox(slide6, 3.6, 4.25, 1.5, 0.3, "R2 keywords", 12, TEXT_SUB, bold=True)
add_textbox(slide6, 5.1, 4.25, 1.5, 0.3, "R3 sentence", 12, TEXT_SUB, bold=True)
add_textbox(slide6, 6.3, 4.25, 1.5, 0.3, "提升", 12, TEXT_SUB, bold=True)

data6 = [
    ("Context Precision", 0.380, 0.550, 44.7),
    ("Context Recall", 0.341, 0.485, 42.2),
    ("Faithfulness", 0.600, 0.685, 14.2),
    ("Answer Relevancy", 0.394, 0.524, 33.0),
]
for j, (m, before, after, imp) in enumerate(data6):
    y = 4.6 + j * 0.38
    add_textbox(slide6, 1.1, y, 2.5, 0.35, m, 13, TEXT_WHITE)
    add_textbox(slide6, 3.6, y, 1.2, 0.35, f"{before:.3f}", 13, TEXT_SUB)
    add_textbox(slide6, 5.1, y, 1.2, 0.35, f"{after:.3f}", 13, ACCENT_GREEN, bold=True)
    add_textbox(slide6, 6.3, y, 1.2, 0.35, f"+{imp:.0f}%", 13, ACCENT_GREEN, bold=True)

add_textbox(slide6, 1.1, 6.3, 6, 0.3, "推論 3.3 s/題  |  GPU 8,860 MB（零額外成本）", 11, TEXT_SUB)

# 失敗範例
add_card(slide6, 8.3, 1.4, 4.3, 5.5)
add_textbox(slide6, 8.5, 1.55, 4.0, 0.35, "仍存在的問題", 14, ACCENT_YELLOW, bold=True)
add_multiline(slide6, 8.5, 1.95, 4.0, 5.0, [
    ("Q：飛機票可以買商務艙嗎？", TEXT_WHITE, True),
    ("標準答案：一律購買經濟艙，", ACCENT_GREEN, False),
    ("12 小時以上可升豪華經濟艙", ACCENT_GREEN, False),
    ("", TEXT_WHITE, False),
    ("模型回答：根據高鐵搭乘規範，", ACCENT_RED, False),
    ("禁止搭乘商務車廂", ACCENT_RED, False),
    ("", TEXT_WHITE, False),
    ("檢索到「高鐵」而非「飛機」規定", TEXT_WHITE, False),
    ("Vector Search 將「商務艙」和", TEXT_WHITE, False),
    ("「商務車廂」語義混淆", TEXT_WHITE, False),
    ("", TEXT_WHITE, False),
    ("診斷：純 Vector Search 缺乏", ACCENT_YELLOW, False),
    ("精確關鍵字匹配能力", ACCENT_YELLOW, False),
    ("", TEXT_WHITE, False),
    ("→ 下一步：加入 Reranker + BM25", ACCENT_BLUE, True),
], font_size=11)


# ============================================================
# P7: Reranker + BM25 Hybrid Search
# ============================================================
slide7 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide7)

add_textbox(slide7, 0.8, 0.4, 10, 0.6, "改良 ③ Reranker + BM25 Hybrid Search", 28, TEXT_WHITE, bold=True)
add_textbox(slide7, 0.8, 0.95, 10, 0.4, "Round 3 → Round 5 → Round 6  |  二階段檢索 + 關鍵字/語義雙路合併", 14, TEXT_SUB)

add_card(slide7, 0.8, 1.4, 7.2, 2.2)
add_textbox(slide7, 1.1, 1.55, 6.8, 0.35, "兩項改動", 16, ACCENT_BLUE, bold=True)
add_multiline(slide7, 1.1, 1.95, 6.8, 1.5, [
    ("A. Reranker（bge-reranker-v2-m3）：先撈 top 9 → 用 CrossEncoder 重排 → top 3", TEXT_WHITE, False),
    ("   精細排序，比 embedding 相似度更準確", TEXT_SUB, False),
    ("B. BM25 Hybrid：Vector(語義) + BM25(關鍵字) → Reciprocal Rank Fusion 合併", TEXT_WHITE, False),
    ("   BM25 對法條編號、專有名詞（如「全勤獎金」「第十二條」）精確匹配更強", TEXT_SUB, False),
], font_size=12)

# 指標
add_card(slide7, 0.8, 3.9, 7.2, 3.0)
add_textbox(slide7, 1.1, 4.05, 6.8, 0.35, "指標對比", 16, ACCENT_GREEN, bold=True)

add_textbox(slide7, 1.1, 4.45, 2.5, 0.3, "指標", 12, TEXT_SUB, bold=True)
add_textbox(slide7, 3.6, 4.45, 1.2, 0.3, "R3", 12, TEXT_SUB, bold=True)
add_textbox(slide7, 4.8, 4.45, 1.2, 0.3, "R5+Reranker", 11, TEXT_SUB, bold=True)
add_textbox(slide7, 6.0, 4.45, 1.2, 0.3, "R6+BM25", 11, TEXT_SUB, bold=True)
add_textbox(slide7, 7.0, 4.45, 1.0, 0.3, "總提升", 12, TEXT_SUB, bold=True)

data7 = [
    ("Context Precision", 0.550, 0.573, 0.650, 18.2),
    ("Context Recall", 0.485, 0.506, 0.601, 23.9),
    ("Faithfulness", 0.685, 0.683, 0.743, 8.5),
    ("Answer Relevancy", 0.524, 0.537, 0.629, 20.0),
]
for j, (m, r3, r5, r6, imp) in enumerate(data7):
    y = 4.8 + j * 0.38
    add_textbox(slide7, 1.1, y, 2.5, 0.35, m, 13, TEXT_WHITE)
    add_textbox(slide7, 3.6, y, 1.0, 0.35, f"{r3:.3f}", 13, TEXT_SUB)
    add_textbox(slide7, 4.8, y, 1.0, 0.35, f"{r5:.3f}", 13, TEXT_SUB)
    add_textbox(slide7, 6.0, y, 1.0, 0.35, f"{r6:.3f}", 13, ACCENT_GREEN, bold=True)
    add_textbox(slide7, 7.0, y, 1.0, 0.35, f"+{imp:.0f}%", 13, ACCENT_GREEN, bold=True)

add_textbox(slide7, 1.1, 6.4, 6, 0.3, "推論 3.8 s/題  |  GPU 10,324 MB（+1.5GB for Reranker，BM25 跑 CPU 不佔 GPU）", 11, TEXT_SUB)

# 失敗範例
add_card(slide7, 8.3, 1.4, 4.3, 5.5)
add_textbox(slide7, 8.5, 1.55, 4.0, 0.35, "仍存在的問題", 14, ACCENT_YELLOW, bold=True)
add_multiline(slide7, 8.5, 1.95, 4.0, 5.0, [
    ("Q：薪水會自動調漲嗎？", TEXT_WHITE, True),
    ("標準答案：不會自動調漲，", ACCENT_GREEN, False),
    ("每年 4 月統一檢視", ACCENT_GREEN, False),
    ("", TEXT_WHITE, False),
    ("模型回答：根據調薪機制...你的", ACCENT_RED, False),
    ("薪水將會自動調漲 ← 幻覺！", ACCENT_RED, False),
    ("", TEXT_WHITE, False),
    ("context 明確寫「4 月底統一調薪」", TEXT_WHITE, False),
    ("但 Llama3 從正確 context 中", TEXT_WHITE, False),
    ("讀出了錯誤結論", TEXT_WHITE, False),
    ("", TEXT_WHITE, False),
    ("診斷：LLM 閱讀理解力不足，", ACCENT_YELLOW, False),
    ("需改善 context 品質讓答案更明顯", ACCENT_YELLOW, False),
    ("", TEXT_WHITE, False),
    ("→ 下一步：加入 HyDE", ACCENT_BLUE, True),
], font_size=11)


# ============================================================
# P8: HyDE
# ============================================================
slide8 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide8)

add_textbox(slide8, 0.8, 0.4, 10, 0.6, "改良 ④ HyDE — Hypothetical Document Embedding", 28, TEXT_WHITE, bold=True)
add_textbox(slide8, 0.8, 0.95, 10, 0.4, "Round 6 → Round 7  |  用假設性回答改善 Vector Search 語義匹配", 14, TEXT_SUB)

add_card(slide8, 0.8, 1.4, 7.2, 2.5)
add_textbox(slide8, 1.1, 1.55, 6.8, 0.35, "核心概念", 16, ACCENT_BLUE, bold=True)
add_multiline(slide8, 1.1, 1.95, 6.8, 1.8, [
    ("問題：用戶口語 vs 法規書面語之間存在語義落差", TEXT_WHITE, False),
    ("", TEXT_WHITE, False),
    ("解法：讓 LLM 先「假裝回答」→ 用假答案做 Vector Search", TEXT_WHITE, False),
    ("  用戶問：「阿公過世可以請幾天喪假？」", TEXT_SUB, False),
    ("  → HyDE 假答案：「依據公司請假辦法，員工祖父母喪亡，給予喪假六日...」", ACCENT_PURPLE, False),
    ("  → 用這段法規語氣的文字去做 Vector Search → 更容易命中喪假條文", TEXT_SUB, False),
], font_size=12)

# 指標
add_card(slide8, 0.8, 4.2, 7.2, 2.8)
add_textbox(slide8, 1.1, 4.35, 6.8, 0.35, "指標對比", 16, ACCENT_GREEN, bold=True)

add_textbox(slide8, 1.1, 4.75, 2.5, 0.3, "指標", 12, TEXT_SUB, bold=True)
add_textbox(slide8, 3.6, 4.75, 1.5, 0.3, "R6 (無 HyDE)", 12, TEXT_SUB, bold=True)
add_textbox(slide8, 5.1, 4.75, 1.5, 0.3, "R7 (+ HyDE)", 12, TEXT_SUB, bold=True)
add_textbox(slide8, 6.3, 4.75, 1.5, 0.3, "提升", 12, TEXT_SUB, bold=True)

data8 = [
    ("Context Precision", 0.650, 0.661, 1.7),
    ("Context Recall", 0.601, 0.615, 2.3),
    ("Faithfulness", 0.743, 0.772, 3.9),
    ("Answer Relevancy", 0.629, 0.665, 5.7),
]
for j, (m, before, after, imp) in enumerate(data8):
    y = 5.1 + j * 0.38
    add_textbox(slide8, 1.1, y, 2.5, 0.35, m, 13, TEXT_WHITE)
    add_textbox(slide8, 3.6, y, 1.2, 0.35, f"{before:.3f}", 13, TEXT_SUB)
    add_textbox(slide8, 5.1, y, 1.2, 0.35, f"{after:.3f}", 13, ACCENT_GREEN, bold=True)
    add_textbox(slide8, 6.3, y, 1.2, 0.35, f"+{imp:.1f}%", 13, ACCENT_GREEN, bold=True)

add_textbox(slide8, 1.1, 6.65, 6, 0.3, "推論 7.4 s/題（+3.6s for HyDE）  |  GPU 10,326 MB（不變）", 11, TEXT_SUB)

# 改善案例
add_card(slide8, 8.3, 1.4, 4.3, 5.5)
add_textbox(slide8, 8.5, 1.55, 4.0, 0.35, "HyDE 改善案例", 14, ACCENT_GREEN, bold=True)
add_multiline(slide8, 8.5, 1.95, 4.0, 5.0, [
    ("Q：薪水會自動調漲嗎？", TEXT_WHITE, True),
    ("", TEXT_WHITE, False),
    ("R6（無 HyDE）：", TEXT_SUB, False),
    ("「薪水將會自動調漲」← 幻覺", ACCENT_RED, False),
    ("", TEXT_WHITE, False),
    ("R7（+ HyDE）：", TEXT_SUB, False),
    ("「不會自動調漲，每年 4 月", ACCENT_GREEN, False),
    (" 統一檢視」← 正確！", ACCENT_GREEN, False),
    ("", TEXT_WHITE, False),
    ("HyDE 假答案語氣更接近法規，", TEXT_WHITE, False),
    ("Vector Search 命中更精準的 chunk，", TEXT_WHITE, False),
    ("LLM 從中讀出正確結論", TEXT_WHITE, False),
    ("", TEXT_WHITE, False),
    ("→ 下一步：Chunk 內容增強", ACCENT_BLUE, True),
], font_size=11)


# ============================================================
# P9: Chunk 增強 + 最終配置
# ============================================================
slide9 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide9)

add_textbox(slide9, 0.8, 0.4, 10, 0.6, "改良 ⑤ Chunk 內容增強 — 標題前綴", 28, TEXT_WHITE, bold=True)
add_textbox(slide9, 0.8, 0.95, 10, 0.4, "Round 7 → Round 8  |  MarkdownHeaderTextSplitter vs MarkdownNodeParser", 14, TEXT_SUB)

add_card(slide9, 0.8, 1.4, 5.5, 2.0)
add_textbox(slide9, 1.1, 1.55, 5.0, 0.35, "改動", 16, ACCENT_BLUE, bold=True)
add_multiline(slide9, 1.1, 1.95, 5.0, 1.3, [
    ("在 chunk content 前面加上結構化標題前綴：", TEXT_WHITE, False),
    ("【員工請假管理辦法 > 假別規定 > 第3條 事假】", ACCENT_PURPLE, False),
    ("→ 原始 .md 不動，僅在 embedding 時拼接", TEXT_SUB, False),
], font_size=12)

# v3 vs v4 對比
add_card(slide9, 6.8, 1.4, 5.7, 2.0)
add_textbox(slide9, 7.1, 1.55, 5.2, 0.35, "兩種 Splitter 對比", 16, ACCENT_BLUE, bold=True)
add_multiline(slide9, 7.1, 1.95, 5.2, 1.3, [
    ("v3 (HeaderSplitter)：三級前綴（文件>章>條）← 勝出", ACCENT_GREEN, False),
    ("v4 (NodeParser)    ：兩級前綴（文件>章）  ← 資訊不足", ACCENT_RED, False),
    ("結論：metadata 完整性比切分引擎更重要", TEXT_SUB, False),
], font_size=12)

# 指標
add_card(slide9, 0.8, 3.7, 5.5, 3.0)
add_textbox(slide9, 1.1, 3.85, 5.0, 0.35, "最終配置指標", 16, ACCENT_GREEN, bold=True)

add_textbox(slide9, 1.1, 4.25, 2.5, 0.3, "指標", 12, TEXT_SUB, bold=True)
add_textbox(slide9, 3.6, 4.25, 1.2, 0.3, "R7 (v2)", 12, TEXT_SUB, bold=True)
add_textbox(slide9, 4.8, 4.25, 1.2, 0.3, "R8a (v3)", 12, TEXT_SUB, bold=True)
add_textbox(slide9, 5.8, 4.25, 1.0, 0.3, "提升", 12, TEXT_SUB, bold=True)

data9 = [
    ("Context Precision", 0.661, 0.668, 1.1),
    ("Context Recall", 0.615, 0.622, 1.1),
    ("Faithfulness", 0.772, 0.778, 0.8),
    ("Answer Relevancy", 0.665, 0.673, 1.2),
]
for j, (m, before, after, imp) in enumerate(data9):
    y = 4.6 + j * 0.38
    add_textbox(slide9, 1.1, y, 2.5, 0.35, m, 13, TEXT_WHITE)
    add_textbox(slide9, 3.6, y, 1.0, 0.35, f"{before:.3f}", 13, TEXT_SUB)
    add_textbox(slide9, 4.8, y, 1.0, 0.35, f"{after:.3f}", 13, ACCENT_GREEN, bold=True)
    add_textbox(slide9, 5.8, y, 1.0, 0.35, f"+{imp:.1f}%", 13, ACCENT_GREEN)

add_textbox(slide9, 1.1, 6.3, 5, 0.3, "推論 8.3 s/題  |  GPU 12,436 MB / 24 GB (52%)", 11, TEXT_SUB)

# 改善案例
add_card(slide9, 6.8, 3.7, 5.7, 3.0)
add_textbox(slide9, 7.1, 3.85, 5.2, 0.35, "標題前綴改善案例", 14, ACCENT_GREEN, bold=True)
add_multiline(slide9, 7.1, 4.25, 5.2, 2.3, [
    ("Q：在公司用自己筆電挖比特幣，會被開除嗎？", TEXT_WHITE, True),
    ("", TEXT_WHITE, False),
    ("R7（無前綴）：搜到辦公場域安全（桌面淨空）", ACCENT_RED, False),
    ("→ 回答模糊：「可能違反規範」", ACCENT_RED, False),
    ("", TEXT_WHITE, False),
    ("R8a（有前綴）：搜到【資訊安全管理規範 >", ACCENT_GREEN, False),
    ("第 5 條 終端設備防護】→「嚴禁加密貨幣挖礦程式」", ACCENT_GREEN, False),
    ("", TEXT_WHITE, False),
    ("原因：前綴「終端設備防護」讓 BM25 精確匹配", TEXT_SUB, False),
], font_size=11)


# ============================================================
# P10: 結論
# ============================================================
slide10 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide10)

add_textbox(slide10, 0.8, 0.4, 10, 0.6, "結論與後續方向", 28, TEXT_WHITE, bold=True)

# 總提升
add_card(slide10, 0.8, 1.2, 6.0, 2.8)
add_textbox(slide10, 1.1, 1.35, 5.5, 0.4, "從原始模型到最終配置的總提升", 16, ACCENT_GREEN, bold=True)

final_data = [
    ("Context Precision", 0.193, 0.668, 246),
    ("Context Recall", 0.159, 0.622, 291),
    ("Faithfulness", 0.517, 0.778, 50),
    ("Answer Relevancy", 0.264, 0.673, 155),
]
add_textbox(slide10, 1.1, 1.8, 2.2, 0.3, "指標", 12, TEXT_SUB, bold=True)
add_textbox(slide10, 3.3, 1.8, 1.0, 0.3, "原始", 12, TEXT_SUB, bold=True)
add_textbox(slide10, 4.3, 1.8, 1.0, 0.3, "最終", 12, TEXT_SUB, bold=True)
add_textbox(slide10, 5.3, 1.8, 1.2, 0.3, "提升", 12, TEXT_SUB, bold=True)

for j, (m, orig, final, imp) in enumerate(final_data):
    y = 2.15 + j * 0.38
    add_textbox(slide10, 1.1, y, 2.2, 0.35, m, 13, TEXT_WHITE)
    add_textbox(slide10, 3.3, y, 1.0, 0.35, f"{orig:.3f}", 13, ACCENT_RED)
    add_textbox(slide10, 4.3, y, 1.0, 0.35, f"{final:.3f}", 13, ACCENT_GREEN, bold=True)
    add_textbox(slide10, 5.3, y, 1.2, 0.35, f"+{imp}%", 14, ACCENT_GREEN, bold=True)

# 關鍵洞察
add_card(slide10, 0.8, 4.3, 6.0, 2.8)
add_textbox(slide10, 1.1, 4.45, 5.5, 0.35, "關鍵洞察", 16, ACCENT_BLUE, bold=True)
add_multiline(slide10, 1.1, 4.85, 5.5, 2.2, [
    ("前 80% 的提升來自 3 個正確的架構決策：", TEXT_WHITE, True),
    ("  1. 選對 Embedding 模型（bge-m3 for 中文）", TEXT_SUB, False),
    ("  2. 選對 Query Rewrite 策略（sentence > keywords）", TEXT_SUB, False),
    ("  3. 混合檢索（Vector + BM25 互補）", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("後 20% 需要精細調校，且邊際遞減", TEXT_WHITE, False),
    ("硬體成本控制良好：GPU 用量 52%，仍有充裕空間", ACCENT_YELLOW, False),
], font_size=12)

# 後續改良方向
add_card(slide10, 7.2, 1.2, 5.3, 5.9)
add_textbox(slide10, 7.5, 1.35, 4.8, 0.4, "後續改良方向", 16, ACCENT_PURPLE, bold=True)
add_multiline(slide10, 7.5, 1.8, 4.8, 5.0, [
    ("短期（不換模型）", ACCENT_BLUE, True),
    ("• Multi-Query Retrieval：多角度查詢合併", TEXT_WHITE, False),
    ("  → 解決跨主題問題的檢索盲區", TEXT_SUB, False),
    ("• Negative Few-Shot Prompt", TEXT_WHITE, False),
    ("  → 教 LLM 判斷何時回答「規章未說明」", TEXT_SUB, False),
    ("• 條文級子 Chunk 拆分", TEXT_WHITE, False),
    ("  → 提升長條文的檢索精準度", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("中期（換模型）", ACCENT_YELLOW, True),
    ("• Qwen2.5-7B / GLM-4-9B（中文原生模型）", TEXT_WHITE, False),
    ("  → 根本提升中文理解力和指令遵循度", TEXT_SUB, False),
    ("  → 預估 Faithfulness 可突破 0.85", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("長期（架構升級）", ACCENT_GREEN, True),
    ("• BGE-M3 原生 Dense+Sparse hybrid", TEXT_WHITE, False),
    ("• Fine-tuning（LoRA/QLoRA on L4）", TEXT_WHITE, False),
    ("• 多模態：YOLO 機房監控整合", TEXT_WHITE, False),
], font_size=11)


# ============================================================
# 儲存
# ============================================================
output_path = "07_evaluation_results/LLM_Local_Deploy_Presentation_v1_0326.pptx"
prs.save(output_path)
print(f"✅ 簡報已產出：{output_path}")
