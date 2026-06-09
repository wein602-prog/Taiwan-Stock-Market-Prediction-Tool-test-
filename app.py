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
from snownlp import SnowNLP
import requests
import holidays
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 0. Streamlit 網頁基本設定
# =========================================================
st.set_page_config(page_title="台股 NLP 多因子預測系統", page_icon="📈", layout="wide")
st.title("📈 台股 NLP 多因子 AI 預測系統")
st.markdown("---")

# ---------------------------------------------------------
# 🛠️ 解決 Matplotlib 中文顯示問題 (使用 st.cache_resource 避免重複下載)
# ---------------------------------------------------------
@st.cache_resource
def setup_font():
    import matplotlib.font_manager as fm
    import os
    font_url = 'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf'
    font_path = 'NotoSansCJKtc-Regular.otf'

    if not os.path.exists(font_path):
        response = requests.get(font_url)
        with open(font_path, 'wb') as f:
            f.write(response.content)

    fm.fontManager.addfont(font_path)
    custom_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = custom_font.get_name() 
    plt.rcParams['axes.unicode_minus'] = False 

setup_font()

# =========================================================
# 左側邊欄 (輸入設定區)
# =========================================================
st.sidebar.header("⚙️ 參數設定")
st.sidebar.markdown("請輸入台灣股票代號。上市股票請加 `.TW`，上櫃股票請加 `.TWO`。")

ticker_symbol = st.sidebar.text_input("股票代號 (例如: 2330.TW, 3711.TW)", value="3711.TW")
run_button = st.sidebar.button("🚀 開始分析", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 **操作提示**\n\n輸入代號後，點擊上方「開始分析」按鈕，系統將自動抓取兩年期數據、籌碼與新聞情緒進行 AI 預測。")

# =========================================================
# 核心運算區塊 (按下按鈕後才會執行)
# =========================================================
if run_button:
    ticker_symbol = ticker_symbol.strip().upper()

    with st.spinner('🔄 正在啟動 NLP 多因子量化引擎，下載 {} 兩年期大數據與訓練模型中，請稍候...'.format(ticker_symbol)):
        
        # 1. 標的設定與基本面資料抓取 (加入防擋 IP 偽裝)
        yf_session = requests.Session()
        yf_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

        ticker = yf.Ticker(ticker_symbol, session=yf_session)
        stock_data = ticker.history(period="2y").reset_index()
        
        if stock_data.empty:
            st.error("❌ 找不到 {} 的歷史股價資料，請確認股票代號是否輸入正確。".format(ticker_symbol))
            st.stop()

        stock_data['Date'] = stock_data['Date'].dt.tz_localize(None).dt.normalize()
        current_price = stock_data['Close'].iloc[-1] # 取得最新收盤價

        stock_id = ticker_symbol.replace(".TW", "").replace(".TWO", "")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get("https://tw.stock.yahoo.com/quote/{}".format(stock_id), headers=headers, timeout=5)
            title_text = res.text[res.text.find('<title>') + 7 : res.text.find('</title>')]
            chinese_name = title_text.split('(')[0].strip()
            display_name = "{} {}".format(ticker_symbol, chinese_name) if "Yahoo" not in chinese_name else ticker_symbol
        except:
            display_name = ticker_symbol

        # 2. 獲取法人籌碼 (FinMind API)
        start_date_chip = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        url = "https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id={}&start_date={}".format(stock_id, start_date_chip)
        try:
            r = requests.get(url, timeout=10)
            chip_data = r.json()
            if chip_data.get('msg') == 'success' and len(chip_data.get('data', [])) > 0:
                df_chips = pd.DataFrame(chip_data['data'])
                df_chips['net_buy'] = (df_chips['buy'] - df_chips['sell']) / 1000 
                df_chips['date'] = pd.to_datetime(df_chips['date'])
                daily_chips = df_chips.groupby('date')['net_buy'].sum().reset_index()
                daily_chips.rename(columns={'date': 'Date', 'net_buy': 'Net_Buy_K'}, inplace=True)
            else:
                daily_chips = pd.DataFrame(columns=['Date', 'Net_Buy_K'])
        except:
            daily_chips = pd.DataFrame(columns=['Date', 'Net_Buy_K'])

        df_merged = pd.merge(stock_data, daily_chips, on='Date', how='left')
        df_merged['Net_Buy_K'] = df_merged['Net_Buy_K'].fillna(0)

        # 3. NLP 新聞情緒分析 (SnowNLP)
        recent_news_sentiment = 0.5 
        news_display_text = []

        try:
            google_news = GNews(language='zh-Hant', country='TW', max_results=5)
            search_keyword = "{} 股票".format(chinese_name if 'chinese_name' in locals() else stock_id)
            news_items = google_news.get_news(search_keyword)
            
            if news_items:
                sentiment_scores = []
                for i, news in enumerate(news_items, 1):
                    title = news.get('title', '')
                    publisher = news.get('publisher', {}).get('title', '未知')
                    
                    s = SnowNLP(title)
                    score = s.sentiments 
                    sentiment_scores.append(score)
                    
                    if score > 0.65: emotion = "🟢 利多"
                    elif score < 0.35: emotion = "🔴 利空"
                    else: emotion = "⚖️ 中性"
                    
                    news_display_text.append("{}. [{}] {} \n ➥ NLP 判定: {} (分數: {:.2f})".format(i, publisher, title, emotion, score))
                
                recent_news_sentiment = np.mean(sentiment_scores)
            else:
                news_display_text.append("⚠️ 目前找不到相關的最新中文新聞。")
        except Exception as e:
            news_display_text.append("⚠️ NLP 新聞模組發生異常。")

        df_merged['Sentiment'] = 0.5 + (df_merged['Close'].pct_change().fillna(0) * 2) + (df_merged['Net_Buy_K'] / 10000)
        df_merged['Sentiment'] = df_merged['Sentiment'].clip(0, 1) 
        df_merged.iloc[-1, df_merged.columns.get_loc('Sentiment')] = recent_news_sentiment 

        # 4. Prophet 模型訓練
        df_prophet = df_merged[['Date', 'Close', 'Net_Buy_K', 'Sentiment']].rename(columns={'Date': 'ds', 'Close': 'y'})

        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=0.15, changepoint_range=0.98)
        model.add_regressor('Net_Buy_K')
        model.add_regressor('Sentiment')
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=30)
        future = future[future['ds'].dt.weekday < 5] 

        last_net_buy = df_prophet['Net_Buy_K'].iloc[-1]
        last_sentiment = df_prophet['Sentiment'].iloc[-1]

        future = pd.merge(future, df_prophet[['ds', 'Net_Buy_K', 'Sentiment']], on='ds', how='left')
        future['Net_Buy_K'] = future['Net_Buy_K'].fillna(last_net_buy)
        future['Sentiment'] = future['Sentiment'].fillna(last_sentiment)

        forecast = model.predict(future)

    st.success("✅ {} 資料載入與模型訓練完成！".format(display_name))

    # =========================================================
    # 網頁視覺化輸出區塊
    # =========================================================

    st.subheader("📊 NLP 多因子 AI 預測圖表")

    fig1 = model.plot(forecast, figsize=(12, 6))
    ax = fig1.gca()

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    black_dot = mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10, label='歷史實際收盤價')
    blue_line = mlines.Line2D([], [], color='#0072B2', linewidth=2, label='AI 預測趨勢 (融合情緒與籌碼)')
    light_blue_patch = mpatches.Patch(color='#0072B2', alpha=0.2, label='信賴區間')
    ax.legend(handles=[black_dot, blue_line, light_blue_patch], loc='best', fontsize=11)

    plt.title('{} 股價 NLP 多因子 AI 預測'.format(display_name), fontsize=14, fontweight='bold')
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('股價', fontsize=12)
    ax.grid(which='major', color='gray', linestyle='-', alpha=0.4)
    ax.grid(which='minor', color='gray', linestyle=':', alpha=0.15)
    plt.tight_layout()

    st.pyplot(fig1)

    history_last_date = df_prophet['ds'].max()
    future_predictions = forecast[forecast['ds'] > history_last_date].copy()

    st.markdown("---")
    st.subheader("📄 決策指揮中心 (分析基準: {})".format(history_last_date.strftime('%Y-%m-%d')))

    # 🌟 修正版：基本面評估 (手動精算避免 API 錯誤)
    st.markdown("#### 💰 基本面評估")
    info = ticker.info
    eps = info.get('trailingEPS', 0)
    pe_ratio = (current_price / eps) if (eps and eps > 0) else info.get('trailingPE', 0)
    book_value = info.get('bookValue', 0)
    pb_ratio = (current_price / book_value) if (book_value and book_value > 0) else info.get('priceToBook', 0)

    try:
        divs = ticker.dividends
        if not divs.empty:
            divs.index = divs.index.tz_localize(None)
            recent_divs = divs[divs.index > (datetime.now() - timedelta(days=365))]
            total_dividend = recent_divs.sum()
            div_yield = (total_dividend / current_price) * 100
        else:
            raw_yield = info.get('dividendYield')
            div_yield = (raw_yield * 100) if raw_yield else 0.0
    except:
        div_yield = 0.0

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("最新收盤價", "{:.2f} 元".format(current_price))
    col_b.metric("本益比 (PE)", "{:.2f} 倍".format(pe_ratio) if pe_ratio else "N/A")
    col_c.metric("淨值比 (PB)", "{:.2f} 倍".format(pb_ratio) if pb_ratio else "N/A")
    col_d.metric("預估殖利率", "{:.2f} %".format(div_yield) if div_yield else "N/A")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧠 NLP 自然語言情緒解析")
        st.info("🚩 **綜合市場情緒分數：{:.2f}** (0=極度恐慌, 1=極度貪婪)".format(recent_news_sentiment))
        for text in news_display_text:
            st.caption(text)

    with col2:
        st.markdown("#### 📈 NLP 籌碼多因子未來推演")
        day_mapping = {0: '週一', 1: '週二', 2: '週三', 3: '週四', 4: '週五'}
        current_year = datetime.now().year
        tw_holidays = holidays.TW(years=[current_year, current_year + 1]) 

        valid_days_count = 0
        first_price = None
        last_price = None

        if not future_predictions.empty:
            for idx, row in future_predictions.iterrows():
                if valid_days_count >= 5: 
                    break
                    
                current_date = row['ds']
                date_str = current_date.strftime('%Y-%m-%d')
                weekday = current_date.weekday()
                weekday_str = day_mapping[weekday]
                
                if current_date in tw_holidays:
                    st.write("📅 {} ({}) | 🛑 **今日休市**".format(date_str, weekday_str))
                    continue
                    
                if first_price is None:
                    first_price = row['yhat']
                last_price = row['yhat']
                
                st.write("📅 **{}** ({}) | 期望價: **${:.2f}** | 區間: ${:.2f} ~ ${:.2f}".format(
                    date_str, weekday_str, row['yhat'], row['yhat_lower'], row['yhat_upper']))
                valid_days_count += 1

    st.markdown("---")
    st.subheader("💡 多因子綜合行動建議")

    is_sentiment_good = recent_news_sentiment > 0.55
    is_trend_up = last_price > first_price if (last_price and first_price) else False 
    is_chip_good = last_net_buy > 0 

    st.write("📌 **當前模型參數狀態：**")
    st.write("1. NLP 新聞情緒：{}".format('**樂觀** 🟢' if is_sentiment_good else '**悲觀 / 觀望** 🔴'))
    st.write("2. 法人籌碼動向：{}".format('**買超** 🟢' if is_chip_good else '**賣超** 🔴'))
    st.write("3. AI 短期預測：{}".format('**趨勢向上** 🟢' if is_trend_up else '**趨勢向下** 🔴'))

    st.markdown("#### 🎯 最終建議：")
    if is_sentiment_good and is_trend_up and is_chip_good:
        st.success("🔥 **【利多共振 - 積極做多】**\n\n情緒、籌碼與時間序列皆偏多，資金處於順風期。")
    elif not is_sentiment_good and is_trend_up and is_chip_good:
        st.warning("⚡ **【籌碼硬扛 - 短線偏多】**\n\n雖然新聞面有雜音，但法人持續買進，模型判定技術面足以支撐上漲。")
    elif is_sentiment_good and not is_trend_up and not is_chip_good:
        st.error("⚠️ **【利多出盡 - 觀望回檔】**\n\n新聞雖好，但法人正在倒貨（拉高出貨），AI 預測即將下彎，請勿追高。")
    elif not is_sentiment_good and not is_trend_up and not is_chip_good:
        st.error("❄️ **【弱勢空頭格局 - 嚴控風險】**\n\n情緒低落且籌碼渙散，建議保持空手。")
    else:
        st.info("⚖️ **【多空分歧 - 區間震盪】**\n\n指標發生衝突，目前缺乏明確方向，建議縮小部位或回歸基本面存股。")

else:
    st.info("👈 請在左側輸入欲查詢的股票代號（例如：2330.TW），並點擊「開始分析」！")
