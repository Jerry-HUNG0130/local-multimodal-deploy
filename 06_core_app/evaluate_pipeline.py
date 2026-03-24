import os
import json
import glob
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# -- Matplotlib Chinese Font --
font_path = 'NotoSansTC.otf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Config
# ============================================================
API_URL = "http://127.0.0.1:8000/api/chat"
DATASET_FILE = "golden_dataset.json"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "07_evaluation_results")

class EvaluationScore(BaseModel):
    context_precision: float = Field(description="Score between 0.0 and 1.0. Is the retrieved context relevant to the question?")
    context_recall: float = Field(description="Score between 0.0 and 1.0. Does the retrieved context contain the information needed to answer the question (compare with Ground Truth)?")
    faithfulness: float = Field(description="Score between 0.0 and 1.0. Is the model's answer factually derived from the retrieved context (no hallucination)?")
    answer_relevancy: float = Field(description="Score between 0.0 and 1.0. How well does the answer address the question directly?")
    reasoning: str = Field(description="Brief explanation of the scores given.")

# ============================================================
# Step 1: Generate Answers from Local RAG
# ============================================================
def generate_answers(test_cases):
    print("🤖 Step 1: 正在讓本地 RAG 模型進行作答...")
    results = []

    for i, case in enumerate(test_cases):
        q = case["question"]
        print(f"  [{i+1}/{len(test_cases)}] {q}")
        try:
            response = requests.post(API_URL, json={"question": q}, timeout=60)
            response.raise_for_status()
            data = response.json()
            results.append({
                "question": q,
                "ground_truth": case["ground_truth"],
                "answer": data.get("answer", ""),
                "context": data.get("context_used", "")
            })
        except Exception as e:
            print(f"  ❌ API 呼叫失敗: {e}")
            results.append({
                "question": q,
                "ground_truth": case["ground_truth"],
                "answer": f"[ERROR] {e}",
                "context": ""
            })

    print(f"✅ Step 1 完成：共 {len(results)} 筆作答結果\n")
    return results

# ============================================================
# Step 2: Evaluate with Gemini
# ============================================================
def load_regulations(lib_dir="../01_docs_library"):
    regulations = []
    for filepath in glob.glob(os.path.join(lib_dir, "*.md")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            filename = os.path.basename(filepath)
            regulations.append(f"--- File: {filename} ---\n{content}\n")
    return "\n".join(regulations)

def evaluate_with_gemini(client, regulations, case):
    prompt = f"""
You are a strict and professional evaluator for a RAG (Retrieval-Augmented Generation) system.
You will evaluate the system's performance on a specific question based on company regulations.

### Company Regulations:
{regulations}

### Evaluation Data:
Question: {case['question']}
Ground Truth: {case['ground_truth']}
Retrieved Context by RAG: {case.get('context', '')}
Model's Answer: {case['answer']}

### Instructions:
Evaluate the model's answer based on the retrieved context and ground truth. Output 4 metric scores (0.0 to 1.0) and a brief reasoning in traditional Chinese.
Be extremely strict. If the answer contradicts the regulations or ground truth, scores should be low.
"""
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EvaluationScore,
            temperature=0.0,
        ),
    )
    return response.text

def evaluate_answers(answer_results):
    load_dotenv(find_dotenv())
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ 錯誤：找不到 GOOGLE_API_KEY，請確認 .env 檔案設定！")
        return None

    client = genai.Client(api_key=api_key)

    print("📖 正在載入公司規章...")
    regulations_text = load_regulations()

    print("👨‍🏫 Step 2: 正在呼叫 Gemini 2.5 Flash 進行嚴格閱卷...")
    evaluated_results = []

    for i, case in enumerate(answer_results):
        print(f"  [{i+1}/{len(answer_results)}] {case['question']}")
        try:
            result_json = evaluate_with_gemini(client, regulations_text, case)
            score_data = json.loads(result_json)
            combined = {**case, **score_data}
            evaluated_results.append(combined)
            print(f"      → {score_data.get('reasoning', '')[:60]}...")
        except Exception as e:
            print(f"  ❌ 評估失敗: {e}")

    if not evaluated_results:
        print("❌ 沒有成功的評估結果！")
        return None

    print(f"✅ Step 2 完成：共 {len(evaluated_results)} 筆評分結果\n")
    return pd.DataFrame(evaluated_results)

# ============================================================
# Step 3: Visualization
# ============================================================
def generate_visual_report(df, output_path="rag_evaluation_report.png", run_label="Baseline"):
    print(f"📊 正在生成視覺化圖表 ({output_path})...")
    metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    metric_labels = ['Context\nPrecision', 'Context\nRecall', 'Faithfulness', 'Answer\nRelevancy']

    avg_scores = [df[m].mean() for m in metrics]
    n = len(df)

    palette = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444']
    bg_color = '#0f172a'
    card_color = '#1e293b'
    text_color = '#f8fafc'
    sub_text = '#94a3b8'
    grid_color = '#334155'

    fig = plt.figure(figsize=(20, 12), facecolor=bg_color)
    fig.suptitle(f'RAG Evaluation Report  —  {run_label}  ({n} Questions)',
                 fontsize=20, fontweight='bold', color=text_color, y=0.97)

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30,
                          left=0.06, right=0.96, top=0.90, bottom=0.07)

    # Panel 1: Radar Chart
    ax_radar = fig.add_subplot(gs[0, 0], polar=True, facecolor=card_color)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    scores = avg_scores + [avg_scores[0]]
    angles_closed = angles + [angles[0]]

    ax_radar.fill(angles_closed, scores, color='#3b82f6', alpha=0.25)
    ax_radar.plot(angles_closed, scores, color='#60a5fa', linewidth=2.5, marker='o', markersize=7)
    for a, s in zip(angles, avg_scores):
        ax_radar.text(a, s + 0.08, f'{s:.2f}', ha='center', va='bottom',
                      fontsize=11, fontweight='bold', color='#60a5fa')
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(metric_labels, fontsize=10, color=text_color)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color=sub_text)
    ax_radar.spines['polar'].set_color(grid_color)
    ax_radar.grid(color=grid_color, linewidth=0.5)
    ax_radar.set_title('Overall Averages', fontsize=13, fontweight='bold',
                       color=text_color, pad=18)

    # Panel 2: Violin + Strip
    ax_violin = fig.add_subplot(gs[0, 1], facecolor=card_color)
    parts = ax_violin.violinplot(
        [df[m].values for m in metrics],
        positions=range(len(metrics)), showmeans=True, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette[i])
        pc.set_alpha(0.35)
    parts['cmeans'].set_color('#fbbf24')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('#f8fafc')
    parts['cmedians'].set_linewidth(1.5)

    for i, m in enumerate(metrics):
        vals = df[m].values
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
        ax_violin.scatter(np.full_like(vals, i) + jitter, vals,
                          color=palette[i], alpha=0.5, s=14, edgecolors='none', zorder=3)

    ax_violin.set_xticks(range(len(metrics)))
    ax_violin.set_xticklabels(metric_labels, fontsize=10, color=text_color)
    ax_violin.set_ylim(-0.05, 1.15)
    ax_violin.set_ylabel('Score', fontsize=11, color=text_color)
    ax_violin.tick_params(colors=sub_text)
    ax_violin.set_facecolor(card_color)
    for spine in ax_violin.spines.values():
        spine.set_color(grid_color)
    ax_violin.grid(axis='y', color=grid_color, linewidth=0.5, alpha=0.5)
    ax_violin.set_title('Score Distribution (Violin + Strip)', fontsize=13,
                        fontweight='bold', color=text_color, pad=12)

    # Panel 3: Summary Table
    ax_table = fig.add_subplot(gs[0, 2], facecolor=card_color)
    ax_table.axis('off')
    stats_data = []
    for m, label in zip(metrics, ['Ctx Prec', 'Ctx Recall', 'Faithful', 'Ans Rel']):
        vals = df[m]
        stats_data.append([
            label,
            f'{vals.mean():.3f}',
            f'{vals.median():.3f}',
            f'{vals.std():.3f}',
            f'{vals.min():.2f}',
            f'{vals.max():.2f}',
            f'{(vals >= 0.8).sum()}/{n}'
        ])
    col_labels = ['Metric', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Pass\n(>=0.8)']
    table = ax_table.table(cellText=stats_data, colLabels=col_labels,
                           loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(grid_color)
        if row == 0:
            cell.set_facecolor('#334155')
            cell.set_text_props(color=text_color, fontweight='bold')
        else:
            cell.set_facecolor(card_color)
            cell.set_text_props(color=text_color)
    ax_table.set_title('Summary Statistics', fontsize=13, fontweight='bold',
                       color=text_color, pad=12)

    # Panel 4: Heatmap (full width)
    ax_heat = fig.add_subplot(gs[1, :], facecolor=card_color)
    df_plot = df.copy()
    df_plot['avg_score'] = df_plot[metrics].mean(axis=1)
    df_plot = df_plot.sort_values('avg_score', ascending=True).reset_index(drop=True)

    heat_data = df_plot[metrics].values.T
    im = ax_heat.imshow(heat_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                        interpolation='nearest')
    ax_heat.set_yticks(range(len(metrics)))
    ax_heat.set_yticklabels(metric_labels, fontsize=10, color=text_color)

    tick_step = max(1, n // 20)
    xtick_pos = list(range(0, n, tick_step))
    ax_heat.set_xticks(xtick_pos)
    ax_heat.set_xticklabels([str(i + 1) for i in xtick_pos], fontsize=8, color=sub_text)
    ax_heat.set_xlabel('Questions (sorted by average score, low → high)', fontsize=11, color=text_color)
    ax_heat.tick_params(colors=sub_text)
    for spine in ax_heat.spines.values():
        spine.set_color(grid_color)
    ax_heat.set_title(f'Per-Question Score Heatmap (all {n} questions)', fontsize=13,
                      fontweight='bold', color=text_color, pad=12)

    cbar = fig.colorbar(im, ax=ax_heat, orientation='vertical', fraction=0.012, pad=0.015)
    cbar.set_label('Score', color=text_color, fontsize=10)
    cbar.ax.tick_params(colors=sub_text)
    cbar.outline.set_edgecolor(grid_color)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"✅ 視覺化圖表已儲存：{output_path}")

# ============================================================
# Main Pipeline
# ============================================================
def main():
    run_label = os.environ.get("RUN_LABEL", "baseline")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load golden dataset
    if not os.path.exists(DATASET_FILE):
        print(f"❌ 找不到 {DATASET_FILE}！")
        return
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"📋 載入 {len(test_cases)} 題測試資料\n")

    # Step 1: Generate answers
    answer_results = generate_answers(test_cases)

    # Step 2: Evaluate with Gemini
    df = evaluate_answers(answer_results)
    if df is None:
        return

    # Print score summary
    print("📊 【RAG 系統評估成績單】")
    metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    display_df = df[['question'] + metrics]
    print(display_df.to_markdown(index=False))

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{run_label}_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 評分數據已儲存：{csv_path}")

    # Step 3: Generate visual report
    png_path = os.path.join(RESULTS_DIR, f"{run_label}_{timestamp}.png")
    generate_visual_report(df, output_path=png_path, run_label=run_label)

if __name__ == "__main__":
    main()
