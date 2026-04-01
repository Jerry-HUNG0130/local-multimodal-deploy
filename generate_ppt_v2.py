#!/usr/bin/env python3
"""
面試用簡報產生器 — LLM 本地端部署專案
第二版 (0401) — 10 分鐘簡報，改良過程精簡為 4 頁
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ============================================================
# 色彩主題（深色背景）
# ============================================================
BG_DARK = RGBColor(0x0F, 0x17, 0x2A)
CARD_BG = RGBColor(0x1E, 0x29, 0x3B)
TEXT_WHITE = RGBColor(0xF8, 0xFA, 0xFC)
TEXT_SUB = RGBColor(0x94, 0xA3, 0xB8)
ACCENT_BLUE = RGBColor(0x3B, 0x82, 0xF6)
ACCENT_GREEN = RGBColor(0x10, 0xB9, 0x81)
ACCENT_RED = RGBColor(0xEF, 0x44, 0x44)
ACCENT_YELLOW = RGBColor(0xF5, 0x9E, 0x0B)
ACCENT_PURPLE = RGBColor(0xA7, 0x8B, 0xFA)
BORDER_COLOR = RGBColor(0x33, 0x41, 0x55)

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


def add_table(slide, left, top, width, height, data, col_widths=None):
    """data = [[cell, ...], ...] 第一行為 header"""
    rows = len(data)
    cols = len(data[0])
    table_shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height))
    table = table_shape.table

    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)

    for r_idx, row in enumerate(data):
        for c_idx, cell_text in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = str(cell_text)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(11)
                paragraph.font.name = "Microsoft JhengHei"
                if r_idx == 0:
                    paragraph.font.bold = True
                    paragraph.font.color.rgb = TEXT_WHITE
                else:
                    paragraph.font.color.rgb = TEXT_WHITE
                paragraph.alignment = PP_ALIGN.CENTER

            # 背景色
            fill = cell.fill
            fill.solid()
            if r_idx == 0:
                fill.fore_color.rgb = RGBColor(0x33, 0x41, 0x55)
            else:
                fill.fore_color.rgb = CARD_BG
    return table_shape


# ============================================================
# P1: 封面
# ============================================================
slide1 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide1)

line_shape = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1), Inches(2.8), Inches(1.5), Inches(0.06))
line_shape.fill.solid()
line_shape.fill.fore_color.rgb = ACCENT_BLUE
line_shape.line.fill.background()

add_textbox(slide1, 1, 3.0, 11, 1.0, "LLM 本地端部署", 44, TEXT_WHITE, bold=True)
add_textbox(slide1, 1, 4.0, 11, 0.8, "全離線 RAG 系統的迭代優化與評測", 22, TEXT_SUB)
add_textbox(slide1, 1, 5.2, 11, 0.5,
            "Qwen2.5-7B  |  BAAI/bge-m3  |  ChromaDB  |  GCP L4 GPU", 16, ACCENT_BLUE)
add_textbox(slide1, 1, 6.5, 11, 0.4, "2026.04", 14, TEXT_SUB)


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
    ("LLM：Qwen2.5-7B-Instruct（Ollama 本地推論）", TEXT_SUB, False),
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
    ("共完成 12 輪迭代、24 組測試", ACCENT_GREEN, True),
], font_size=14)


# ============================================================
# P3: RAG 四大評估指標
# ============================================================
slide3 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide3)

add_textbox(slide3, 0.8, 0.4, 10, 0.6, "RAG 四大評估指標", 28, TEXT_WHITE, bold=True)
add_textbox(slide3, 0.8, 0.95, 10, 0.4,
            "Based on RAGAS: Automated Evaluation of Retrieval Augmented Generation", 12, TEXT_SUB)

metrics_info = [
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

for i, (name_en, name_zh, meaning, example, ideal, color) in enumerate(metrics_info):
    col = i % 2
    row = i // 2
    x = 0.8 + col * 6.2
    y = 1.5 + row * 2.9

    add_card(slide3, x, y, 5.8, 2.6)
    add_textbox(slide3, x + 0.2, y + 0.15, 3.5, 0.4, name_en, 16, color, bold=True)
    add_textbox(slide3, x + 0.2, y + 0.55, 3.5, 0.3, name_zh, 12, TEXT_SUB)
    add_textbox(slide3, x + 0.2, y + 0.9, 5.3, 0.6, meaning, 11, TEXT_WHITE)
    add_textbox(slide3, x + 0.2, y + 1.55, 5.3, 0.7, example, 10, TEXT_SUB)
    add_textbox(slide3, x + 3.8, y + 0.15, 1.8, 0.35, f"業界標準 {ideal}", 11, ACCENT_GREEN)


# ============================================================
# P4: 原生模型表現 + 問題診斷
# ============================================================
slide4 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide4)

add_textbox(slide4, 0.8, 0.4, 10, 0.6, "原生模型表現 — Llama3 8B + mxbai-embed", 28, TEXT_WHITE, bold=True)

# 指標卡片
add_card(slide4, 0.8, 1.2, 5.5, 2.8)
add_textbox(slide4, 1.1, 1.35, 5, 0.4, "原始模型指標", 16, ACCENT_RED, bold=True)

add_table(slide4, 1.1, 1.85, 5.0, 2.0, [
    ["指標", "數值", "業界標準"],
    ["Context Precision", "0.193", "≥ 0.70"],
    ["Context Recall", "0.159", "≥ 0.70"],
    ["Faithfulness", "0.517", "≥ 0.85"],
    ["Answer Relevancy", "0.264", "≥ 0.70"],
], col_widths=[2.2, 1.4, 1.4])

add_textbox(slide4, 1.1, 3.6, 5.0, 0.3, "推論速度 4.0 s/題  |  GPU 9,731 MB", 11, TEXT_SUB)

# 失敗範例 ①
add_card(slide4, 6.8, 1.2, 5.7, 1.6)
add_textbox(slide4, 7.1, 1.3, 5.2, 0.35, "失敗範例 ① 檢索完全錯誤", 14, ACCENT_RED, bold=True)
add_multiline(slide4, 7.1, 1.7, 5.2, 1.0, [
    ("Q：密碼設定有什麼複雜度要求？", TEXT_WHITE, False),
    ("模型回答：根據《員工績效考核辦法》第 4 條 遲到...", ACCENT_RED, False),
    ("診斷：mxbai embedding 語義差，問密碼搜到績效考核", TEXT_SUB, False),
], font_size=11)

# 失敗範例 ②
add_card(slide4, 6.8, 3.0, 5.7, 1.6)
add_textbox(slide4, 7.1, 3.1, 5.2, 0.35, "失敗範例 ② 嚴重語義漂移", 14, ACCENT_RED, bold=True)
add_multiline(slide4, 7.1, 3.5, 5.2, 1.0, [
    ("Q：幫同事代打卡被抓到，最嚴重懲處？", TEXT_WHITE, False),
    ("模型回答：根據《員工績效考核辦法》... 獎金發放、薪資調整...", ACCENT_RED, False),
    ("診斷：搜到教育訓練而非考勤制度 → 需更換 Embedding 模型", TEXT_SUB, False),
], font_size=11)

# 問題總結
add_card(slide4, 0.8, 4.3, 11.7, 2.7)
add_textbox(slide4, 1.1, 4.45, 11.2, 0.4, "瓶頸分析 → 四大改良方向", 16, ACCENT_YELLOW, bold=True)
add_multiline(slide4, 1.1, 4.95, 5.3, 2.0, [
    ("① Embedding 語義能力不足", TEXT_WHITE, True),
    ("   mxbai → BAAI/bge-m3 (GPU，多語言)", TEXT_SUB, False),
    ("② Chunking 跨條文污染", TEXT_WHITE, True),
    ("   固定 500 字 → 按 Markdown 結構切分", TEXT_SUB, False),
], font_size=12)
add_multiline(slide4, 6.8, 4.95, 5.3, 2.0, [
    ("③ Query Rewrite 破壞語義", TEXT_WHITE, True),
    ("   關鍵字拆散 → 完整句子改寫", TEXT_SUB, False),
    ("④ LLM 中文理解力不足", TEXT_WHITE, True),
    ("   Llama3 8B → Qwen2.5-7B（原生中文）", TEXT_SUB, False),
], font_size=12)


# ============================================================
# P5: 改良① Retrieval Pipeline 優化（R1→R8a 打包）
# ============================================================
slide5 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide5)

add_textbox(slide5, 0.8, 0.4, 12, 0.6,
            "改良 ① Retrieval Pipeline 優化", 28, TEXT_WHITE, bold=True)
add_textbox(slide5, 0.8, 0.95, 12, 0.3,
            "六項改動打包：Embedding + Chunking + Query Rewrite + BM25 Hybrid + Reranker + HyDE", 13, TEXT_SUB)

# 左側：六項改動說明
add_card(slide5, 0.8, 1.5, 6.0, 5.3)
add_textbox(slide5, 1.1, 1.6, 5.5, 0.4, "逐步堆疊的改動", 16, ACCENT_BLUE, bold=True)

add_multiline(slide5, 1.1, 2.1, 5.5, 4.5, [
    ("❶ Embedding：mxbai → bge-m3 (GPU)", TEXT_WHITE, True),
    ("   繁中語義理解翻倍，四項指標 +56~124%", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("❷ Chunking：固定 500 字 → Markdown 結構切分", TEXT_WHITE, True),
    ("   每個 chunk = 一條法規，含三級 metadata", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("❸ Query Rewrite：keywords → sentence", TEXT_WHITE, True),
    ("   保留完整語義，CP/CR 各提升 +42~45%", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("❹ BM25 Hybrid：語義 + 關鍵字雙路檢索", TEXT_WHITE, True),
    ("   RRF 合併，四項指標全面 +13~19%", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("❺ Reranker：bge-reranker-v2-m3 二階段重排", TEXT_WHITE, True),
    ("   top 9 → CrossEncoder 精排 → top 3", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("❻ HyDE：假設性回答改善 Vector Search", TEXT_WHITE, True),
    ("   縮小口語 vs 法規書面語的語義落差", TEXT_SUB, False),
], font_size=12, line_spacing=1.1)

# 右側上：指標對比表
add_card(slide5, 7.2, 1.5, 5.3, 2.6)
add_textbox(slide5, 7.5, 1.6, 4.8, 0.35, "階段性指標對比", 14, ACCENT_GREEN, bold=True)

add_table(slide5, 7.5, 2.05, 4.8, 2.0, [
    ["指標", "原始", "R3", "R6", "R8a"],
    ["CP", "0.193", "0.550", "0.650", "0.668"],
    ["CR", "0.159", "0.485", "0.601", "0.622"],
    ["Faith", "0.517", "0.685", "0.743", "0.778"],
    ["AR", "0.264", "0.524", "0.629", "0.673"],
    ["s/題", "4.0", "3.3", "3.8", "8.3"],
], col_widths=[0.8, 0.9, 0.9, 0.9, 0.9])

# 右側下：檢索流程圖
add_card(slide5, 7.2, 4.3, 5.3, 2.5)
add_textbox(slide5, 7.5, 4.4, 4.8, 0.35, "R8a 完整檢索流程", 14, ACCENT_BLUE, bold=True)
add_multiline(slide5, 7.5, 4.85, 4.8, 1.8, [
    ("用戶口語問題", TEXT_WHITE, False),
    ("  ↓  Sentence Rewrite（正式書面語）", ACCENT_BLUE, False),
    ("  ↓  HyDE（LLM 產生假設性回答）", ACCENT_PURPLE, False),
    ("  ↓  Vector Search + BM25 → RRF 合併", ACCENT_YELLOW, False),
    ("  ↓  Reranker 精排 → top 3 context", ACCENT_GREEN, False),
    ("  ↓  LLM 根據 context 生成回答", TEXT_WHITE, True),
], font_size=12, line_spacing=1.15)


# ============================================================
# P6: 改良② LLM 升級 Qwen2.5 + Few-Shot
# ============================================================
slide6 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide6)

add_textbox(slide6, 0.8, 0.4, 12, 0.6,
            "改良 ② LLM 升級 — Llama3 → Qwen2.5-7B", 28, TEXT_WHITE, bold=True)
add_textbox(slide6, 0.8, 0.95, 12, 0.3,
            "Retrieval 天花板已到，進一步提升需從 LLM 層面突破", 13, TEXT_SUB)

# 左上：為什麼換模型
add_card(slide6, 0.8, 1.5, 5.8, 2.3)
add_textbox(slide6, 1.1, 1.6, 5.3, 0.35, "為什麼需要換模型？", 16, ACCENT_YELLOW, bold=True)
add_multiline(slide6, 1.1, 2.05, 5.3, 1.5, [
    ("Llama3 8B 的 Prompt Engineering 天花板已確認：", TEXT_WHITE, False),
    ("• R4 嚴格指令 → Faith 下降（不敢回答正確答案）", TEXT_SUB, False),
    ("• R9 Few-Shot → Faith -7.1%（8K context 被擠壓）", TEXT_SUB, False),
    ("• R10 更好的檢索 → Faith -6.8%（更多 context 反而更混亂）", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("結論：Llama3 的中文閱讀理解力是根本瓶頸", ACCENT_RED, True),
], font_size=12, line_spacing=1.1)

# 左下：模型選擇過程
add_card(slide6, 0.8, 4.0, 5.8, 2.8)
add_textbox(slide6, 1.1, 4.1, 5.3, 0.35, "模型選擇：一次失敗 + 一次成功", 16, ACCENT_BLUE, bold=True)
add_multiline(slide6, 1.1, 4.55, 5.3, 2.0, [
    ("❌ R11 Taiwan-LLM 8B（繁中微調）", ACCENT_RED, True),
    ("   Faith 0.778 → 0.533（-31.5%）", ACCENT_RED, False),
    ("   閱讀理解嚴重退步：看到「禁止」卻答「可以」", TEXT_SUB, False),
    ("   繁中微調方向是「對話」而非「文件問答」", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("✅ R12 Qwen2.5-7B-Instruct（原生中文）", ACCENT_GREEN, True),
    ("   Faith 0.778 → 0.833（+7.1%）", ACCENT_GREEN, False),
    ("   32K context（Llama3 的 4 倍）、推論快 25%", TEXT_SUB, False),
    ("   零幻覺：不確定時正確回答「規章未說明」", TEXT_SUB, False),
], font_size=12, line_spacing=1.1)

# 右上：R12d Few-Shot 的突破
add_card(slide6, 7.0, 1.5, 5.5, 2.3)
add_textbox(slide6, 7.3, 1.6, 5.0, 0.35, "R12d: Few-Shot + Qwen2.5 的突破", 14, ACCENT_GREEN, bold=True)
add_multiline(slide6, 7.3, 2.05, 5.0, 1.5, [
    ("Llama3 few-shot 失敗（8K context 被擠壓）", TEXT_SUB, False),
    ("Qwen2.5 few-shot 成功（32K context 完美消化）", ACCENT_GREEN, False),
    ("", TEXT_WHITE, False),
    ("3 個精準範例：", TEXT_WHITE, True),
    ("① 正確引用條文（25 萬 GPU → CEO 核准）", TEXT_SUB, False),
    ("② 注意排除條款（SRE 不得申請 WFH）", TEXT_SUB, False),
    ("③ 規章未涵蓋時（春節加班費 → 規章未說明）", TEXT_SUB, False),
], font_size=12, line_spacing=1.1)

# 右下：指標對比
add_card(slide6, 7.0, 4.0, 5.5, 2.8)
add_textbox(slide6, 7.3, 4.1, 5.0, 0.35, "R8a (Llama3) → R12d (Qwen2.5) 指標對比", 14, ACCENT_GREEN, bold=True)

add_table(slide6, 7.3, 4.55, 5.0, 1.8, [
    ["指標", "R8a Llama3", "R12d Qwen2.5", "變化"],
    ["CP", "0.668", "0.662", "-0.9%"],
    ["CR", "0.622", "0.615", "-1.1%"],
    ["Faithfulness", "0.778", "0.856", "+10.0%"],
    ["Answer Rel.", "0.673", "0.737", "+9.5%"],
    ["Faith = 0", "1 題", "0 題", "全部通過"],
    ["推論速度", "8.3 s/題", "5.5 s/題", "-34%"],
], col_widths=[1.2, 1.2, 1.3, 0.9])


# ============================================================
# P7: 最終成果 + 關鍵洞察
# ============================================================
slide7 = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide7)

add_textbox(slide7, 0.8, 0.4, 12, 0.6,
            "結論 — 12 輪迭代的最終成果", 28, TEXT_WHITE, bold=True)

# 左上：總提升表
add_card(slide7, 0.8, 1.2, 5.8, 2.8)
add_textbox(slide7, 1.1, 1.3, 5.3, 0.4, "從原始模型到最終配置", 16, ACCENT_GREEN, bold=True)

add_table(slide7, 1.1, 1.8, 5.3, 2.0, [
    ["指標", "原始模型", "R12d 最終", "總提升"],
    ["Context Precision", "0.193", "0.662", "+243%"],
    ["Context Recall", "0.159", "0.615", "+287%"],
    ["Faithfulness", "0.517", "0.856", "+65.6%"],
    ["Answer Relevancy", "0.264", "0.737", "+179%"],
    ["Faith = 0 題數", "—", "0 題", "零幻覺"],
    ["推論速度", "4.0 s/題", "5.5 s/題", "可接受"],
], col_widths=[1.4, 1.1, 1.1, 1.0])

# 左下：硬體成本
add_card(slide7, 0.8, 4.2, 5.8, 2.6)
add_textbox(slide7, 1.1, 4.3, 5.3, 0.4, "硬體成本控制", 16, ACCENT_BLUE, bold=True)
add_multiline(slide7, 1.1, 4.8, 5.3, 2.0, [
    ("GPU 記憶體：9.7 → 15.2 GB（L4 24GB 內仍有餘裕）", TEXT_WHITE, False),
    ("推論速度：4.0 → 5.5 s/題（HyDE 佔主要增量）", TEXT_WHITE, False),
    ("", TEXT_WHITE, False),
    ("如延遲敏感，可關閉 HyDE 回到 ~3 s/題", TEXT_SUB, False),
    ("僅損失 ~2% 各項指標", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("最終配置：v3 + Sentence Rewrite + HyDE +", ACCENT_BLUE, False),
    ("BM25 Hybrid + Reranker + Qwen2.5 + Few-Shot", ACCENT_BLUE, False),
], font_size=12, line_spacing=1.15)

# 右側：關鍵洞察
add_card(slide7, 7.0, 1.2, 5.5, 5.6)
add_textbox(slide7, 7.3, 1.3, 5.0, 0.4, "關鍵洞察", 16, ACCENT_YELLOW, bold=True)

add_multiline(slide7, 7.3, 1.8, 5.0, 5.0, [
    ("投報率最高的 4 項改動（佔總提升 ~90%）", TEXT_WHITE, True),
    ("  1. Embedding 模型選對（bge-m3）— 成本為零", ACCENT_GREEN, False),
    ("  2. Query Rewrite 策略修正 — 成本為零", ACCENT_GREEN, False),
    ("  3. BM25 Hybrid Search — CPU 運算不佔 GPU", ACCENT_GREEN, False),
    ("  4. LLM 升級 Qwen2.5 — 速度反而更快", ACCENT_GREEN, False),
    ("", TEXT_WHITE, False),
    ("失敗嘗試同樣有價值", TEXT_WHITE, True),
    ("  • 嚴格 Prompt → Llama3 不敢答正確答案", TEXT_SUB, False),
    ("  • Few-Shot + Llama3 → 學格式不學邏輯", TEXT_SUB, False),
    ("  • Taiwan-LLM → 繁中微調 ≠ RAG 能力", TEXT_SUB, False),
    ("  • Query Decompose → 更好檢索反而更混亂", TEXT_SUB, False),
    ("", TEXT_WHITE, False),
    ("核心結論", TEXT_WHITE, True),
    ("  「選對模型 + 選對策略」比「極致調參」重要", ACCENT_YELLOW, False),
    ("  前 80% 提升來自 3 個正確的架構決策", ACCENT_YELLOW, False),
    ("  後 20% 需要精細調校，且邊際遞減", TEXT_SUB, False),
], font_size=12, line_spacing=1.15)


# ============================================================
# 儲存
# ============================================================
output_path = "07_evaluation_results/LLM_Local_Deploy_Presentation_v2_0401.pptx"
prs.save(output_path)
print(f"✅ 簡報已儲存：{output_path}")
print(f"   共 {len(prs.slides)} 頁")
