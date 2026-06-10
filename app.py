# ==========================================
# 台股 NLP 輔助分析與 Prompt 自動生成器 (修正版)
# ==========================================

import ipywidgets as widgets
from IPython.display import display, clear_output
import requests
from bs4 import BeautifulSoup

# --- 1. 設計 App 互動介面 (UI) ---
title_html = widgets.HTML(value="<h2>📈 台股 AI 分析指令生成器 (防失真版)</h2><p>請輸入代碼、今日收盤價與奇摩股市網址，系統將自動擷取資訊並生成嚴謹的 AI 分析指令。</p>")

stock_input = widgets.Text(
    description="股票代碼:",
    placeholder="例如: 3711 (日月光投控)",
    layout=widgets.Layout(width='50%')
)

price_input = widgets.Text(
    description="今日收盤價:",
    placeholder="例如: 539.00 (重要！防止預測脫鉤)",
    layout=widgets.Layout(width='50%')
)

url_input = widgets.Text(
    description="奇摩網址:",
    placeholder="請貼上奇摩股市該檔股票的網址",
    layout=widgets.Layout(width='80%')
)

analyze_btn = widgets.Button(
    description="產生分析指令",
    button_style='success',
    icon='magic'
)

output_area = widgets.Output()

# --- 2. 爬蟲與資料擷取邏輯 ---
def fetch_yahoo_finance(url):
    # 設定 User-Agent 避免被網站直接阻擋
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 抓取網頁標題
        page_title = soup.find('title').text if soup.find('title') else "無法取得標題"
        
        # 抓取頁面中的主要新聞敘述與文字
        paragraphs = soup.find_all('p', limit=5)
        text_content = "\n".join([p.text for p in paragraphs if p.text])
        
        return f"【網頁標題】: {page_title}\n【近期市場敘述】:\n{text_content}"
        
    except Exception as e:
        return f"網址資料抓取失敗。錯誤訊息: {e}"

# --- 3. 按鈕點擊後的執行邏輯 ---
def on_button_click(b):
    with output_area:
        clear_output()
        stock = stock_input.value.strip()
        price = price_input.value.strip()
        url = url_input.value.strip()
        
        if not stock or not url or not price:
            print("⚠️ 錯誤：請確認已完整輸入「股票代碼」、「今日收盤價」與「奇摩網址」。")
            return
            
        print(f"⏳ 正在前往擷取 {stock} 的近期資訊，請稍候...\n")
        
        # 呼叫爬蟲函數
        scraped_data = fetch_yahoo_finance(url)
        
        # 組合給 Claude 的系統提示詞 (已加入防呆與價格錨定機制)
        prompt = f"""
請扮演資深量化交易員與台股分析師。我將提供你一檔股票的現狀，請為我分析「明日開盤的看漲/看跌走向與勝率」。

【分析目標與基準數據】
* 股票代碼/名稱：{stock}
* **今日真實收盤價：{price} 元 (這是你所有預測的絕對基準點)**

【近期市場資訊 (來自網路擷取)】
{scraped_data}

【你的分析任務與嚴格規範】
1. **新聞情緒辨識：** 根據上述資訊，簡述目前市場氛圍。若新聞偏向極度樂觀（利多），請務必加入「是否為高檔出貨文」的反向思考。
2. **防脫鉤預測機制：** 你的明日價格預測**必須**以今日收盤價 ({price} 元) 為起點計算漲跌幅，絕對不可憑空產生一個偏離今日收盤價的預期數字。
3. **漲跌幅防呆檢查：** 台股每日最大漲跌幅限制為 10%。你給出的明日期望價格區間，最高不可超過 {float(price) * 1.10:.2f} 元，最低不可低於 {float(price) * 0.90:.2f} 元。
4. **輸出結論：** 綜合以上，給出明日預期走向（跳空上漲 / 震盪走高 / 區間盤整 / 震盪走低 / 跳空大跌），以及量化的勝率估計（例如 65%）。
5. **操作建議：** 給出具體的短線停損點或停利點建議。

請保持專業、直白，並直接給出分析結果。
"""
        print("✅ 資訊擷取完成！防失真分析指令已生成：\n")
        print("==================================================")
        print("👉 【請複製以下整段文字，貼上給 Claude 進行精準預測】 👈")
        print("==================================================\n")
        print(prompt)

# --- 4. 綁定事件並顯示 App ---
analyze_btn.on_click(on_button_click)
display(title_html, stock_input, price_input, url_input, analyze_btn, output_area)
