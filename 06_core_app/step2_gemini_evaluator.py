import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

import matplotlib.font_manager as fm

# 強制 Matplotlib 使用下載的中文字型，解決豆腐塊問題
font_path = 'NotoSansTC.otf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

class EvaluationScore(BaseModel):
    context_precision: float = Field(description="Score between 0.0 and 1.0. Is the retrieved context relevant to the question?")
    context_recall: float = Field(description="Score between 0.0 and 1.0. Does the retrieved context contain the information needed to answer the question (compare with Ground Truth)?")
    faithfulness: float = Field(description="Score between 0.0 and 1.0. Is the model's answer factually derived from the retrieved context (no hallucination)?")
    answer_relevancy: float = Field(description="Score between 0.0 and 1.0. How well does the answer address the question directly?")
    reasoning: str = Field(description="Brief explanation of the scores given.")

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

def generate_visual_report(df):
    print("📊 正在生成視覺化圖表 (rag_evaluation_report.png)...")
    q_col = 'question'
    metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    
    avg_scores = [df[m].mean() for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Local RAG System Performance Evaluation (Custom Gemini Pipeline)', fontsize=16, fontweight='bold')

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    scores = avg_scores + [avg_scores[0]]
    angles += [angles[0]]
    
    ax1 = plt.subplot(121, polar=True)
    ax1.fill(angles, scores, color='skyblue', alpha=0.4)
    ax1.plot(angles, scores, color='blue', linewidth=2)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.set_title('Overall RAG Capabilities (Average)', pad=20, fontsize=14)

    df_melted = df.melt(id_vars=[q_col], value_vars=metrics, 
                        var_name='Metric', value_name='Score')
    df_melted['Short Question'] = df_melted[q_col].apply(lambda x: x[:12] + '...' if len(x) > 12 else x)
    
    sns.barplot(data=df_melted, x='Short Question', y='Score', hue='Metric', ax=ax2, palette='Set2')
    ax2.set_ylim(0, 1.1)
    ax2.set_title('Score Breakdown by Question', fontsize=14)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("rag_evaluation_report.png", dpi=300, bbox_inches='tight')
    print("✅ 視覺化圖表已成功儲存為：rag_evaluation_report.png")

def main():
    load_dotenv(find_dotenv())
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ 錯誤：找不到 GOOGLE_API_KEY，請確認 .env 檔案設定！")
        return
        
    client = genai.Client(api_key=api_key)
    
    data_file = "evaluation_data.json"
    if not os.path.exists(data_file):
        print(f"❌ 找不到 {data_file}！請先執行 step1_generate_answers.py！")
        return
        
    with open(data_file, "r", encoding="utf-8") as f:
        evaluation_data = json.load(f)
        
    print("📖 正在載入公司規章...")
    regulations_text = load_regulations()
    
    print("\n👨‍🏫 正在呼叫 Gemini 2.5 Flash 進行嚴格閱卷...")
    evaluated_results = []
    
    for i, case in enumerate(evaluation_data):
        print(f"  👉 評估第 {i+1} 題: {case['question']}")
        try:
            result_json = evaluate_with_gemini(client, regulations_text, case)
            score_data = json.loads(result_json)
            
            # 合併原始資料與評分
            combined = {**case, **score_data}
            evaluated_results.append(combined)
            print(f"      [評分理由] {score_data.get('reasoning', '')}")
        except Exception as e:
            print(f"  ❌ 評估失敗: {e}")
            
    if not evaluated_results:
        print("❌ 沒有成功的評估結果！")
        return
        
    df = pd.DataFrame(evaluated_results)
    
    print("\n📊 【RAG 系統評估成績單】")
    display_df = df[['question', 'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']]
    print(display_df.to_markdown(index=False))
    
    generate_visual_report(df)

if __name__ == "__main__":
    main()
