import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from prophet import Prophet
from datetime import datetime, timedelta
from gnews import GNews
import requests
import holidays
import matplotlib.font_manager as fm
import os
import tempfile

# ==========================================
# 網頁基礎設定
# ==========================================
st.set_page_config(page_title="AI 股票決策指揮中心", page_icon="📈", layout="centered")

# ------------------------------------------
# 🌟 招財/大展鴻圖 視覺設計 (CSS 注入)
# ------------------------------------------
def set_wealth_background():
    # 使用帶有金色/股市意象的背景圖，並疊加一層深色半透明遮罩(rgba)，確保白字依然清晰可見
    # 右下角加入「大展鴻圖」的浮水印
    page_bg_css = """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(17, 24, 39, 0.85), rgba(17, 24, 39, 0.85)), url("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    .stApp::after {
        content: '大展鴻圖 💰 招財進寶';
        position: fixed;
        bottom: 20px;
        right: 20px;
        font-size: 28px;
        color: rgba(255, 215, 0, 0.15); /* 淡淡的土豪金 */
        font-weight: bold;
        z-index: 100;
        pointer-events: none; /* 確保浮水印不會阻擋滑鼠點擊 */
        letter-spacing: 2px;
    }
    </style>
    """
    st.markdown(page_bg_css, unsafe_allow_html=True)

set_wealth_background()

st.title("📈 AI 股票預測與決策指揮中心")

# ==========================================
# 📖 系統使用指南與說明 (折疊面板)
# ==========================================
with st.expander("ℹ️ 系統使用指南與說明 (點擊展開)", expanded=False):
    st.markdown("""
    **歡迎使用 AI 股票決策指揮中心！** 本系統整合技術面、基本面、籌碼面與消息面，為您提供全方位的投資參考。
    
    ### 📌 操作步驟
    1. **輸入代號**：在左側設定區輸入您想查詢的台股代號（上市股票請加 `.TW`，上櫃請加 `.TWO`，例如：台積電 `2330.TW`、台新金 `2887.TW`）。
    2. **開始分析**：點擊「🚀 開始分析」按鈕，系統將即時抓取最新數據並啟動 AI 運算模型。
    
    ### 📊 報告區塊說明
    * **AI 趨勢預測圖**：藍線為 AI 推演的未來 30 天可能走勢，淺藍色區塊為波動的信賴區間。
    * **決策指揮中心報告**：為您快速掃描「總經大環境（大盤/美債/半導體）」、「基本面體質（本益比/殖利率）」與「籌碼動能（三大法人/大戶資金流向）」。
    * **AI 操盤總結與策略建議**：系統會自動根據上述各項指標進行「四大維度」計分，並給出直接、簡潔的戰鬥指令（從強勢做多到嚴控風險共 5 種層級）。
    * **近期新聞與情緒判定**：自動抓取標的最新新聞，並透過專屬關鍵字演算法，判定消息面為「🟢 利多」、「🔴 利空」或「⚪ 中性」。
    
    > ⚠️ **免責聲明**：本系統之 AI 預測與策略建議僅供學術研究與參考之用，不構成任何實質買賣建議。金融市場變幻莫測，投資人應自行謹慎評估風險並自負盈虧。
    """)

# ==========================================
# 🛠️ 核心優化：安全寫入中文字型
# ==========================================
@st.cache_resource
def load_font():
    font_url = 'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf'
    font_path = os.path.join(tempfile.gettempdir(), 'NotoSansCJKtc-Regular.otf')
    
    if not os.path.exists(font_path):
        try:
            response = requests.get(font_url, timeout=10)
            with open(font_path, 'wb') as f:
                f.write(response.content)
        except Exception:
            return
            
    try:
        fm.fontManager.addfont(font_path)
        custom_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = custom_font.get_name() 
        plt.rcParams['axes.unicode_minus'] = False 
    except Exception:
        pass

load_font()

# ==========================================
# 📰 新聞情緒分析模組 (NLP Keyword-based)
# ==========================================
def analyze_news_sentiment(text):
    positive_keywords = ['看好', '成長', '創高', '大增', '買超', '利多', '突破', '上漲', '增長', '受惠', '升評', '調升', '雙增', '爆單', '強勁', '新高', '優於預期', '獲利', '配息', '大賺', '飆', '買盤', '利潤', '翻紅']
    negative_keywords = ['看淡', '衰退', '新低', '大減', '賣超', '利空', '跌破', '下跌', '減少', '受害', '降評', '調降', '雙減', '砍單', '疲弱', '不如預期', '虧損', '下修', '拋售', '逃命', '爆雷', '警戒', '外資倒貨']
    
    score = 0
    for kw in positive_keywords:
        if kw in text: score += 1
    for kw in negative_keywords:
        if kw in text: score -= 1
    return score

# ==========================================
# 側邊欄：使用者輸入區與作者資訊
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center;'>🐱 財源廣進 💰</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.header("設定區")
ticker_symbol = st.sidebar.text_input("請輸入股票代號 (例如: 2887.TW)", value="2887.TW")
analyze_button = st.sidebar.button("🚀 開始分析")

# 🎓 在這裡新增了您的學校與作者資訊
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="color: #A0AEC0; font-size: 0.9em; line-height: 1.6;">
    <b>國立高雄大學</b><br>
    電機工程系 在職碩專班(一年級)<br>
    L1145102<br>
    作者：陳韋豪
</div>
""", unsafe_allow_html=True)

# ==========================================
# 主程式運算區塊
# ==========================================
if analyze_button:
    with st.spinner(f"正在連線伺服器，全力運算 {ticker_symbol} 的數據中..."):
        
        # ------------------------------------------
        # 1. 取得歷史股價與中文名稱
        # ------------------------------------------
        ticker = yf.Ticker(ticker_symbol)
        stock_data = ticker.history(period="2y")
        
        if stock_data.empty:
            st.error(f"❌ 無法從 Yahoo Finance 取得 {ticker_symbol} 的股價資料！請確認代號是否正確。")
            st.stop()
            
        stock_id = ticker_symbol.replace(".TW", "").replace(".TWO", "")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
            res = requests.get(url, headers=headers, timeout=5)
            
            start_idx = res.text.find('<title>')
            if start_idx != -1:
                start_idx += 7
                end_idx = res.text.find('</title>')
                chinese_name = res.text[start_idx:end_idx].split('(')[0].strip()
                display_name = f"{ticker_symbol} {chinese_name}" if "Yahoo" not in chinese_name else ticker_symbol
                stock_name_for_news = chinese_name if "Yahoo" not in chinese_name else stock_id
            else:
                display_name, stock_name_for_news = ticker_symbol, stock_id
        except:
            display_name, stock_name_for_news = ticker_symbol, stock_id

        st.success(f"✅ 成功取得標的：【{display_name}】")

        # ------------------------------------------
        # 2. 總體經濟與產業環境評估
        # ------------------------------------------
        macro_tickers = {'^TWII': '台灣加權指數', '^SOX': '費城半導體指數', '^TNX': '美10年期公債殖利率'}
        macro_results, macro_score = {}, 0
        for sym, name in macro_tickers.items():
            try:
                m_data = yf.Ticker(sym).history(period="6mo")['Close']
                if not m_data.empty:
                    curr_val = m_data.iloc[-1]
                    ma60 = m_data.rolling(60).mean().iloc[-1]
                    if sym == '^TNX':
                        is_tailwind = curr_val < ma60
                        macro_score += 1 if is_tailwind else -1
                    else:
                        is_tailwind = curr_val > ma60
                        macro_score += 1 if is_tailwind else -1
                    status = "🟢 偏多/寬鬆" if is_tailwind else "🔴 偏空/緊縮"
                    macro_results[name] = f"目前 {curr_val:.2f} | 季線 {ma60:.2f} ➔ {status}"
            except:
                macro_results[name] = "⚠️ 無法取得"

        env_status = "大環境順風 🌬️ (多頭動能強)" if macro_score >= 2 else "大環境逆風 🌪️ (系統性風險較高)" if macro_score <= -2 else "大環境中性 ⚖️ (震盪整理)"

        # ------------------------------------------
        # 3. 基本面資料
        # ------------------------------------------
        try:
            info = ticker.info
            dividend_yield = info.get('dividendYield', 0)
            trailing_yield = info.get('trailingAnnualDividendYield', 0)
            final_yield = dividend_yield if dividend_yield else trailing_yield
            yield_str = f"{final_yield:.2f}%" if final_yield > 1 else f"{final_yield * 100:.2f}%" if final_yield else "無資料"

            pe_ratio = info.get('trailingPE', None)
            pe_str = f"{pe_ratio:.2f} 倍" if pe_ratio else "無資料"
            eps = info.get('trailingEps', None)
            eps_str = f"{eps:.2f} 元" if eps else "無資料"
            pb_
