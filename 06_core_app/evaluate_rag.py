import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv, find_dotenv
from datasets import Dataset

# --- 🏆 回歸經典穩定版 Import (請直接無視執行時的黃色警告) ---
from ragas import evaluate, RunConfig
from ragas.metrics import (
    Faithfulness,        
    AnswerRelevancy,    
    ContextPrecision,   
    ContextRecall       
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# -----------------------------------------------------------

# 強制 Matplotlib 使用中文字型，解決豆腐塊問題
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

load_dotenv(find_dotenv())
if not os.environ.get("GOOGLE_API_KEY"):
    print("❌ 錯誤：找不到 GOOGLE_API_KEY，請確認 .env 檔案設定！")
    exit()

API_URL = "http://127.0.0.1:8000/api/chat"

def generate_visual_report(df):
    print("📊 正在生成視覺化圖表 (rag_evaluation_report.png)...")
    q_col = 'user_input' if 'user_input' in df.columns else 'question'
    metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    existing_metrics = [m for m in metrics if m in df.columns]
    
    df[existing_metrics] = df[existing_metrics].fillna(0.0)
    avg_scores = [df[m].mean() for m in existing_metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Local RAG System Performance Evaluation (Llama 3 + BGE-M3)', fontsize=16, fontweight='bold')

    angles = np.linspace(0, 2 * np.pi, len(existing_metrics), endpoint=False).tolist()
    scores = avg_scores + [avg_scores[0]]
    angles += [angles[0]]
    
    ax1 = plt.subplot(121, polar=True)
    ax1.fill(angles, scores, color='skyblue', alpha=0.4)
    ax1.plot(angles, scores, color='blue', linewidth=2)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in existing_metrics], fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.set_title('Overall RAG Capabilities (Average)', pad=20, fontsize=14)

    df_melted = df.melt(id_vars=[q_col], value_vars=existing_metrics, 
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
    print("📝 準備『考卷』 (Ground Truth)...")
    test_cases = [
        {
            "question": "我老婆下個月要生小孩了，請問她有幾天產假？我可以請幾天假陪她？",
            "ground_truth": "女性生產享有 8 週產假。男性享有 7 日陪產檢及陪產假。"
        },
        {
            "question": "下禮拜要去高雄出差，想說搭高鐵商務艙比較舒服，到了左營再搭計程車去客戶那，請問費用可以全額報銷嗎？",
            "ground_truth": "高鐵原則上一律搭乘「標準車廂」，禁止擅自搭乘商務車廂。短程交通以大眾運輸為主，除非須載運重型設備、夜間加班超過十點或遭遇颱風豪雨，否則不予核銷計程車費用。"
        },
        {
            "question": "為了趕專案，我可以把開發環境的資料存在我自己的隨身碟，帶回家用自己電腦弄嗎？",
            "ground_truth": "嚴禁使用私人隨身碟或外接硬碟存取公司資料。全公司之電腦預設鎖定 USB 連線功能。"
        },
        {
            "question": "這週五下午想提早回鄉下，我可以申請這拜五遠端辦公 (WFH) 嗎？",
            "ground_truth": "原則上每週一、週五全面禁止 WFH。特殊事由需經總經理特簽。"
        },
        {
            "question": "早上塞車，打卡時間是 09:45，這樣算遲到嗎？會扣多少錢？每個月有容錯空間嗎？",
            "ground_truth": "打卡時間超過 09:31 且未達 10:00 者視為遲到，當日薪資將直接扣發半小時之基準時薪。每月系統提供「2 次、每次 5 分鐘內」之容錯寬限期。"
        },
        {
            "question": "我開發完新功能了，可以直接把 code 合併進 main 分支嗎？自動化測試覆蓋率要達標多少？",
            "ground_truth": "嚴禁直接向 main 分支提交 (Commit) 程式碼。所有提交的程式碼皆須通過自動化的單元測試，且測試覆蓋率不得低於 75%。"
        },
        {
            "question": "如果機房 CCTV 視覺監控發現溫度超過 28 度，第一時間 (T+0 與 T+1) 系統會做什麼動作？",
            "ground_truth": "T+0 系統會自動擷取當前畫面並產生具備時間戳記的影像檔至系統資料庫留存分析。T+1 分鐘會觸發實體警報燈紅色雙向閃爍及蜂鳴聲，同時推送圖文通報到 Slack/Line 頻道並標記值班工程師。"
        },
        {
            "question": "主管要派我去上一個價值六萬塊的 AWS 雲端架構師培訓班，公司全額出錢，我有什麼義務嗎？",
            "ground_truth": "由公司全額補助超過新台幣 50,000 元之單項高階認證與培訓專案，員工必須簽署「培訓服務承諾書」，承諾於結訓取得證照後繼續留任本公司至少一年。若提前離職須依未滿之月份比例退回全額補助款。"
        }
    ]

    questions, answers, contexts, ground_truths = [], [], [], []

    print("🤖 正在讓受測者 (本地 Llama 3) 進行作答...")
    for case in test_cases:
        q = case["question"]
        print(f"  👉 測試問題: {q}")
        try:
            response = requests.post(API_URL, json={"question": q}, timeout=30)
            response_data = response.json()
            
            ans = response_data.get("answer", "")
            ctx = response_data.get("context_used", "")
            
            questions.append(q)
            answers.append(ans)
            contexts.append([ctx] if ctx else [""])
            ground_truths.append(case["ground_truth"])
        except Exception as e:
            print(f"  ❌ API 呼叫失敗: {e}")
            return

    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    })

    print("\n👨‍🏫 正在呼叫 Gemini 2.0 Flash 進行嚴格閱卷...")
    
    from langchain_google_genai import HarmCategory, HarmBlockThreshold
    
    # 宣告大腦，並解除安全封印與強制 JSON 輸出
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
        # 降低安全審查，避免「生小孩」、「機房」等詞彙被誤擋
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    from ragas.embeddings import GoogleEmbeddings
    ragas_llm = LangchainLLMWrapper(gemini_llm)
    ragas_emb = GoogleEmbeddings(model="models/text-embedding-004")
    
    # 設置執行參數避免掛起 (Deadlock) 與無限重試
    # run_config 用於控制並發數量，減少觸發 API 的高並發死鎖
    run_config = RunConfig(max_workers=2, max_retries=2, timeout=60)
    
    result = evaluate(
        dataset=dataset,
        metrics=[
            ContextPrecision(llm=ragas_llm), 
            ContextRecall(llm=ragas_llm), 
            Faithfulness(llm=ragas_llm), 
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)
        ],
        run_config=run_config,
        raise_exceptions=True
    )

    df = result.to_pandas()
    
    print("\n📊 【RAG 系統評估成績單】")
    q_col = 'user_input' if 'user_input' in df.columns else 'question'
    target_metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    
    existing_cols = [q_col] + [m for m in target_metrics if m in df.columns]
    display_df = df[existing_cols]
    display_df = display_df.fillna(0.0)
    print(display_df.to_markdown(index=False))
    
    generate_visual_report(df)

if __name__ == "__main__":
    main()