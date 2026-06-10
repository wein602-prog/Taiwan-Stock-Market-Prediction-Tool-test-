import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# --- 1. 介面設定 ---
st.set_page_config(page_title="台股 AI 分析指令生成器", page_icon="🤖")
st.title("🤖 台股 AI 分析指令生成器 (全自動版)")
st.markdown("只需輸入台股代碼，系統將自動連網抓取**最新收盤價**與**奇摩股市新聞**，並生成防失真的 AI 分析指令。")

stock_input = st.text_input("請輸入股票代碼 (例如: 2330 或 3711):")

# --- 2. 自動抓取股價邏輯 (使用 yfinance) ---
def fetch_current_price(stock_id):
    """自動判斷是上市(.TW)還是上櫃(.TWO)並抓取最新收盤價"""
    try:
        # 先嘗試上市代碼
        ticker = yf.Ticker(f"{stock_id}.TW")
        hist = ticker.history(period="1d")
        if not hist.empty:
            return round(hist['Close'].iloc[-1], 2)
            
        # 若無資料，嘗試上櫃代碼
        ticker = yf.Ticker(f"{stock_id}.TWO")
        hist = ticker.history(period="1d")
        if not hist.empty:
            return round(hist['Close'].iloc[-1], 2)
            
        return None
    except Exception:
        return None

# --- 3. 自動抓取新聞邏輯 (奇摩股市) ---
def auto_fetch_yahoo_news(stock_id):
    """自動組合網址並抓取奇摩股市的近期文字敘述"""
    url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 抓取網頁標題與段落
        page_title = soup.find('title').text if soup.find('title') else "無法取得標題"
        paragraphs = soup.find_all('p', limit=6)
        text_content = "\n".join([p.text for p in paragraphs if p.text and len(p.text) > 10])
        
        if not text_content:
            text_content = "(無明顯新聞文字，請依賴基本面與技術面數據分析)"
            
        return f"【資料來源】: {page_title}\n【近期市場敘述】:\n{text_content}"
    except Exception as e:
        return f"新聞抓取失敗，錯誤訊息: {e}"

# --- 4. 執行按鈕與組合 Prompt ---
if st.button("全自動產生分析指令", type="primary"):
    if not stock_input.strip():
        st.error("⚠️ 請先輸入股票代碼！")
    else:
        stock_id = stock_input.strip()
        
        with st.spinner(f"🌐 正在自動搜尋 {stock_id} 的股價與新聞，請稍候..."):
            
            # 自動抓價格
            current_price = fetch_current_price(stock_id)
            
            if current_price is None:
                st.error(f"⚠️ 找不到代碼 {stock_id} 的真實股價。請確認代碼是否正確（請只輸入純數字）。")
            else:
                # 自動抓新聞
                scraped_data = auto_fetch_yahoo_news(stock_id)
                
                # 計算 10% 防呆區間
                upper_limit = current_price * 1.10
                lower_limit = current_price * 0.90
                
                prompt = f"""
請扮演資深量化交易員與台股分析師。我將提供你一檔股票的現狀，請為我分析「明日開盤的看漲/看跌走向與勝率」。

【分析目標與基準數據】
* 股票代碼：{stock_id}
* **今日真實收盤價：{current_price} 元 (這是你所有預測的絕對基準點)**

【近期市場資訊 (自動擷取自網路)】
{scraped_data}

【你的分析任務與嚴格規範】
1. **新聞情緒辨識：** 根據上述資訊，簡述目前市場氛圍。若新聞偏向極度樂觀（利多），請務必加入「是否為高檔出貨文」的反向思考。
2. **防脫鉤預測機制：** 你的明日價格預測**必須**以今日收盤價 ({current_price} 元) 為起點計算漲跌幅，絕對不可憑空產生一個偏離今日收盤價的預期數字。
3. **漲跌幅防呆檢查：** 台股每日最大漲跌幅限制為 10%。你給出的明日期望價格區間，最高不可超過 {upper_limit:.2f} 元，最低不可低於 {lower_limit:.2f} 元。
4. **輸出結論：** 綜合以上，給出明日預期走向（跳空上漲 / 震盪走高 / 區間盤整 / 震盪走低 / 跳空大跌），以及量化的勝率估計（例如 65%）。
5. **操作建議：** 給出具體的短線停損點或停利點建議。

請保持專業、直白，並直接給出分析結果。
"""
                st.success(f"✅ 抓取成功！最新收盤價為：**{current_price} 元**。分析指令已生成：")
                st.markdown("👉 **【請點擊右上方複製圖示，貼上給 Claude 進行精準預測】** 👈")
                st.code(prompt, language="markdown")
