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
import matplotlib.font_manager as fm
import os

warnings.filterwarnings('ignore')

# =========================================================
# 網頁基本設定
# =========================================================
st.set_page_config(page_title="多因子 AI 預測引擎", page_icon="📈", layout="wide")
st.title("📈 NLP 多因子量化分析引擎")
st.markdown("結合 **Prophet 時間序列**、**法人籌碼** 與 **SnowNLP 新聞情緒** 的台股預測模型。")

# =========================================================
# 🛠️ 解決 Matplotlib 中文顯示問題 (快取處理以提升效能)
# =========================================================
@st.cache_resource
def setup_font():
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
# 側邊欄：使用者參數設定
# =========================================================
with st.sidebar:
    st.header("⚙️ 參數設定")
    ticker_symbol = st.text_input("輸入台股代號 (例如: 2887.TW)", value="2887.TW")
    analyze_button = st.button("🚀 開始分析", use_container_width=True)

# =========================================================
# 主程式邏輯
# =========================================================
if analyze_button:
    with st.spinner(f"🔄 正在啟動 NLP 多因子量化引擎，下載 {ticker_symbol} 兩年期大數據..."):
        
        # 1. 標的設定與基本面資料抓取
        ticker = yf.Ticker(ticker_symbol)
        stock_data = ticker.history(period="2y").reset_index()
        stock_data['Date'] = stock_data['Date'].dt.tz_localize(None).dt.normalize()

        stock_id = ticker_symbol.replace(".TW", "").replace(".TWO", "")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(f"https://tw.stock.yahoo.com/quote/{stock_id}", headers=headers, timeout=5)
            title_text = res.text[res.text.find('<title>') + 7 : res.text.find('</title>')]
            chinese_name = title_text.split('(')[0].strip()
            display_name = f"{ticker_symbol} {chinese_name}" if "Yahoo" not in chinese_name else ticker_symbol
        except:
            display_name = ticker_symbol

    with st.spinner("📊 正在融合三大法人歷史籌碼數據..."):
        # 2. 特徵工程 A：獲取 2 年期歷史法人籌碼 (FinMind)
        start_date_chip = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id={stock_id}&start_date={start_date_chip}"
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

    with st.spinner("📰 正在執行 NLP 模組：分析近期新聞情緒分數..."):
        # 3. 特徵工程 B：NLP 新聞情緒分析 (SnowNLP)
        recent_news_sentiment = 0.5
        news_display_text = []

        try:
            google_news = GNews(language='zh-Hant', country='TW', max_results=5)
            news_items = google_news.get_news(f"{chinese_name if 'chinese_name' in locals() else stock_id} 股票")

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

                    news_display_text.append(f"**{i}. [{publisher}]** {title}  \n➥ *NLP 判定: {emotion} (分數: {score:.2f})*")

                recent_news_sentiment = np.mean(sentiment_scores)
            else:
                news_display_text.append("⚠️ 目前找不到相關的最新中文新聞。")
        except Exception as e:
            news_display_text.append("⚠️ NLP 新聞模組發生異常。")

        df_merged['Sentiment'] = 0.5 + (df_merged['Close'].pct_change().fillna(0) * 2) + (df_merged['Net_Buy_K'] / 10000)
        df_merged['Sentiment'] = df_merged['Sentiment'].clip(0, 1)
        df_merged.iloc[-1, df_merged.columns.get_loc('Sentiment')] = recent_news_sentiment

    with st.spinner("🤖 正在訓練 Prophet 多因子模型推演未來趨勢..."):
        # 4. Prophet 多因子模型訓練
        df_prophet = df_merged[['Date', 'Close', 'Net_Buy_K', 'Sentiment']].rename(columns={'Date': 'ds', 'Close': 'y'})

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.15,
            changepoint_range=0.98
        )

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

    # =========================================================
    # 區塊 A：繪製趨勢主圖表
    # =========================================================
    st.success(f"✅ 分析完成！【{display_name}】多因子 AI 預測報告如下：")
    
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

    plt.title(f'{display_name} 股價 NLP 多因子 AI 預測', fontsize=14, fontweight='bold')
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('股價', fontsize=12)
    ax.grid(which='major', color='gray', linestyle='-', alpha=0.4)
    ax.grid(which='minor', color='gray', linestyle=':', alpha=0.15)
    
    st.pyplot(fig1)

    # =========================================================
    # 區塊 B：文字報告整合
    # =========================================================
    history_last_date = df_prophet['ds'].max()
    future_predictions = forecast[forecast['ds'] > history_last_date].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 NLP 自然語言情緒解析")
        st.write(f"🚩 **綜合市場情緒分數：{recent_news_sentiment:.2f}** *(0=極度恐慌, 1=極度貪婪)*")
        for text in news_display_text:
            st.markdown(text)

    with col2:
        st.subheader("📈 AI 多因子未來推演")
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
                    st.write(f"📅 {date_str} ({weekday_str}) | 🛑 今日休市 ({tw_holidays.get(current_date)})")
                    continue

                if first_price is None:
                    first_price = row['yhat']
                last_price = row['yhat']

                st.markdown(f"📅 **{date_str} ({weekday_str})** | 期望價: **${row['yhat']:.2f}** | 區間: ${row['yhat_lower']:.2f} ~ ${row['yhat_upper']:.2f}")
                valid_days_count += 1

    # =========================================================
    # 區塊 C：總結建議
    # =========================================================
    st.markdown("---")
    st.subheader("💡 多因子綜合行動建議")

    is_sentiment_good = recent_news_sentiment > 0.55
    is_trend_up = last_price > first_price if (last_price and first_price) else False
    is_chip_good = last_net_buy > 0

    st.markdown("📌 **當前模型參數狀態：**")
    st.markdown(f"- NLP 新聞情緒：**{'樂觀 🟢' if is_sentiment_good else '悲觀 / 觀望 🔴'}**")
    st.markdown(f"- 法人籌碼動向：**{'買超 🟢' if is_chip_good else '賣超 🔴'}**")
    st.markdown(f"- AI 短期預測：**{'趨勢向上 🟢' if is_trend_up else '趨勢向下 🔴'}**")

    st.markdown("🎯 **最終建議：**")
    if is_sentiment_good and is_trend_up and is_chip_good:
        st.success("🔥 **【利多共振 - 積極做多】**\n\n情緒、籌碼與時間序列皆偏多，資金處於順風期。")
    elif not is_sentiment_good and is_trend_up and is_chip_good:
        st.info("⚡ **【籌碼硬扛 - 短線偏多】**\n\n雖然新聞面有雜音，但法人持續買進，模型判定技術面足以支撐上漲。")
    elif is_sentiment_good and not is_trend_up and not is_chip_good:
        st.warning("⚠️ **【利多出盡 - 觀望回檔】**\n\n新聞雖好，但法人正在倒貨（拉高出貨），AI 預測即將下彎，請勿追高。")
    elif not is_sentiment_good and not is_trend_up and not is_chip_good:
        st.error("❄️ **【弱勢空頭格局 - 嚴控風險】**\n\n情緒低落且籌碼渙散，建議保持空手。")
    else:
        st.warning("⚖️ **【多空分歧 - 區間震盪】**\n\n指標發生衝突，目前缺乏明確方向，建議縮小部位或回歸基本面存股。")
