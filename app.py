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
    *   **AI 趨勢預測圖**：藍線為 AI 推演的未來 30 天可能走勢，淺藍色區塊為波動的信賴區間。
    *   **決策指揮中心報告**：為您快速掃描「總經大環境（大盤/美債/半導體）」、「基本面體質（本益比/殖利率）」與「籌碼動能（三大法人/大戶資金流向）」。
    *   **AI 操盤總結與策略建議**：系統會自動根據上述各項指標進行「四大維度」計分，並給出直接、簡潔的戰鬥指令（從強勢做多到嚴控風險共 5 種層級）。
    *   **近期新聞與情緒判定**：自動抓取標的最新新聞，並透過專屬關鍵字演算法，判定消息面為「🟢 利多」、「🔴 利空」或「⚪ 中性」。
    
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
# 側邊欄：使用者輸入區
# ==========================================
# 增加招財吉祥物
st.sidebar.markdown("<h2 style='text-align: center;'>🐱 財源廣進 💰</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.header("設定區")
ticker_symbol = st.sidebar.text_input("請輸入股票代號 (例如: 2887.TW)", value="2887.TW")
analyze_button = st.sidebar.button("🚀 開始分析")

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
            pb_ratio = info.get('priceToBook', None)
            pb_str = f"{pb_ratio:.2f} 倍" if pb_ratio else "無資料"
        except Exception:
            yield_str, pe_str, eps_str, pb_str = "⚠️ API限流", "⚠️ API限流", "⚠️ API限流", "⚠️ API限流"

        # ------------------------------------------
        # 4. 三大法人與 OBV 資金動能
        # ------------------------------------------
        chip_text = ""
        try:
            start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
            url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id={stock_id}&start_date={start_date}"
            r = requests.get(url, timeout=5)
            chip_data = r.json()

            if chip_data.get('msg') == 'success' and len(chip_data.get('data', [])) > 0:
                df_chips = pd.DataFrame(chip_data['data'])
                if 'buy' in df_chips.columns and 'sell' in df_chips.columns:
                    df_chips['net_buy'] = (df_chips['buy'] - df_chips['sell']) / 1000
                    recent_date = df_chips['date'].max()
                    df_recent = df_chips[df_chips['date'] == recent_date]

                    foreign = df_recent[df_recent['name'] == 'Foreign_Investor']['net_buy'].sum()
                    trust = df_recent[df_recent['name'] == 'Investment_Trust']['net_buy'].sum()
                    dealer = df_recent[df_recent['name'].str.contains('Dealer')]['net_buy'].sum()

                    chip_text = f"最新 ({recent_date})：外資 **{foreign:,.0f}** 張 | 投信 **{trust:,.0f}** 張 | 自營商 **{dealer:,.0f}** 張"
                else:
                    chip_text = "⚠️ 法人資料格式異動"
            else:
                chip_text = "⚠️ 無法取得三大法人最新數據"
        except:
            chip_text = "⚠️ 籌碼資料連線異常"

        obv = [0]
        if len(stock_data) > 1:
            for i in range(1, len(stock_data)):
                if stock_data['Close'].iloc[i] > stock_data['Close'].iloc[i-1]:
                    obv.append(obv[-1] + stock_data['Volume'].iloc[i])
                elif stock_data['Close'].iloc[i] < stock_data['Close'].iloc[i-1]:
                    obv.append(obv[-1] - stock_data['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
        stock_data['OBV'] = obv
        
        if len(stock_data) >= 5:
            obv_trend = "🟢 資金流入 (大戶偏多)" if stock_data['OBV'].iloc[-1] > stock_data['OBV'].iloc[-5] else "🔴 資金流出 (大戶偏空)"
        else:
            obv_trend = "⚪ 資料不足無法計算"

        # ------------------------------------------
        # 5. 提前抓取新聞並計算情緒分數
        # ------------------------------------------
        news_items_processed = []
        overall_news_score = 0
        try:
            google_news = GNews(language='zh-Hant', country='TW', max_results=5)
            news_items = google_news.get_news(f"{stock_name_for_news} 股票")
            
            if news_items:
                for news in news_items:
                    title = news.get('title', '無標題')
                    desc = news.get('description', '')
                    # 計算單則新聞情緒分數
                    item_score = analyze_news_sentiment(title + " " + desc)
                    overall_news_score += item_score
                    
                    # 賦予情緒標籤
                    if item_score > 0:
                        sentiment_label = "🟢 利多"
                    elif item_score < 0:
                        sentiment_label = "🔴 利空"
                    else:
                        sentiment_label = "⚪ 中性"
                        
                    news_items_processed.append({
                        'title': title,
                        'url': news.get('url', '#'),
                        'publisher': news.get('publisher', {}).get('title', '未知來源'),
                        'date': str(news.get('published date', ''))[:16],
                        'sentiment': sentiment_label
                    })
        except:
            pass # 發生錯誤時分數保持為 0

        # 決定整體新聞市場情緒
        if overall_news_score > 0:
            news_sentiment_status = "偏多 🟢 (市場消息看好)"
        elif overall_news_score < 0:
            news_sentiment_status = "偏空 🔴 (市場消息看淡)"
        else:
            news_sentiment_status = "中性 ⚪ (無明顯利多/空)"

        # ------------------------------------------
        # 6. Prophet AI 預測模型
        # ------------------------------------------
        df = stock_data.reset_index()
        df['Date'] = df['Date'].dt.tz_localize(None)
        df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).dropna()
        
        model = Prophet(
            daily_seasonality=False, 
            weekly_seasonality=True,      
            yearly_seasonality=True,      
            changepoint_prior_scale=0.05, 
            changepoint_range=0.8         
        )
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=30) 
        future = future[future['ds'].dt.weekday < 5] 
        forecast = model.predict(future)


        # =========================================================
        # 🟢 輸出畫面區塊
        # =========================================================
        
        # --- 圖表區 ---
        st.subheader("📊 AI 趨勢預測圖")
        fig1, ax = plt.subplots(figsize=(10, 5))
        model.plot(forecast, ax=ax)
        
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        ax.xaxis.set_minor_locator(mdates.DayLocator())

        black_dot = mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10, label='歷史實際收盤價')
        blue_line = mlines.Line2D([], [], color='#0072B2', linewidth=2, label='AI 預測主要趨勢')
        light_blue_patch = mpatches.Patch(color='#0072B2', alpha=0.2, label='預測信賴區間 / 波動範圍')
        ax.legend(handles=[black_dot, blue_line, light_blue_patch], loc='best', fontsize=9, framealpha=0.9, edgecolor='gray')

        plt.title(f'{display_name} 股價 AI 預測與趨勢分析', fontsize=14, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('股價', fontsize=12)
        ax.grid(which='major', color='gray', linestyle='-', alpha=0.4)
        ax.grid(which='minor', color='gray', linestyle=':', alpha=0.15)
        st.pyplot(fig1)

        # --- 數據報告區 ---
        st.markdown("---")
        st.subheader("📄 決策指揮中心報告")
        
        st.markdown(f"**🌍 總經大環境：** {env_status}")
        for name, status in macro_results.items():
            st.write(f"🔹 {name}: {status}")
            
        st.write("") 
            
        st.markdown("**💰 基本面評估：**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("EPS", eps_str)
        col2.metric("本益比", pe_str)
        col3.metric("淨值比", pb_str)
        col4.metric("殖利率", yield_str)

        st.markdown("**🕵️‍♂️ 籌碼動能：**")
        st.write(f"- OBV 近五日動能：{obv_trend}")
        st.write(f"- 三大法人：{chip_text}")
            
        # --- 未來 5 日推演區 ---
        st.markdown("---")
        st.subheader("📈 AI 未來 5 個有效交易日推演")
        
        history_last_date = df_prophet['ds'].max()
        future_predictions = forecast[forecast['ds'] > history_last_date].copy()
        
        first_price, last_price = None, None
        valid_days = 0
        day_mapping = {0: '週一', 1: '週二', 2: '週三', 3: '週四', 4: '週五', 5: '週六', 6: '週日'}
        
        try:
            tw_holidays = holidays.country_holidays('TW', years=[datetime.now().year, datetime.now().year + 1])
        except:
            tw_holidays = holidays.TW(years=[datetime.now().year, datetime.now().year + 1])
        
        for _, row in future_predictions.iterrows():
            if valid_days >= 5: break
            curr_date = row['ds']
            weekday = curr_date.weekday()
            
            if curr_date.date() in tw_holidays: 
                holiday_name = tw_holidays.get(curr_date.date())
                st.warning(f"📅 **{curr_date.strftime('%Y-%m-%d')} ({day_mapping[weekday]})** | 🛑 今日休市 ({holiday_name})，暫無交易預測")
                continue
            
            if first_price is None: first_price = row['yhat']
            last_price = row['yhat']
            
            st.info(f"📅 **{curr_date.strftime('%Y-%m-%d')} ({day_mapping[weekday]})** | 期望價: **${row['yhat']:.2f}** (區間: ${row['yhat_lower']:.2f} ~ ${row['yhat_upper']:.2f})")
            valid_days += 1

        # --- 策略總結區 ---
        st.markdown("---")
        st.subheader("💡 AI 操盤總結與策略建議")
        
        # 評估四大維度
        is_macro_good = macro_score >= 0 
        is_obv_good = "流入" in obv_trend 
        is_trend_up = last_price > first_price if (last_price and first_price) else False 
        is_news_good = overall_news_score > 0
        
        # 計算多方總分 (滿分 4 分)
        bull_score = sum([is_macro_good, is_obv_good, is_trend_up, is_news_good])
        
        st.write("📌 **當前盤勢結構 (四大維度交叉比對)：**")
        st.write(f"1. 總經大環境：{'偏多 🟢' if is_macro_good else '偏空 / 保守 🔴'}")
        st.write(f"2. 技術與籌碼：{'大戶資金流入 🟢' if is_obv_good else '大戶資金流出 🔴'}")
        st.write(f"3. AI 短期預測：{'趨勢向上 🟢' if is_trend_up else '趨勢向下 🔴'}")
        st.write(f"4. 近期新聞情緒：{news_sentiment_status}")
        
        st.write("🎯 **綜合行動建議：**")
        if bull_score == 4:
            st.success("🔥 **【強勢多頭 - 全面做多】**\n\n**戰況：** 四大指標全面翻多，多方具備絕對優勢。\n\n**行動：** 順勢加碼，抱緊獲利！空手者可於盤中回檔果斷切入。")
        elif bull_score == 3:
            st.info("📈 **【偏多震盪 - 逢低佈局】**\n\n**戰況：** 趨勢偏多但伴隨部分雜訊，上檔仍有空間。\n\n**行動：** 嚴禁追高！採「逢低分批買進」，並嚴設移動停利鎖住獲利。")
        elif bull_score == 2:
            st.warning(f"⚖️ **【多空交戰 - 區間操作】**\n\n**戰況：** 多空勢均力敵，盤勢進入方向選擇的關鍵期。\n\n**行動：** 短線採「高出低進」賺價差。存股族請評估目前殖利率 ({yield_str})，若達標可維持定期定額。")
        elif bull_score == 1:
            st.error("⚠️ **【弱勢探底 - 反彈減碼】**\n\n**戰況：** 賣壓沉重，防線潰退中，籌碼與趨勢轉弱。\n\n**行動：** 嚴禁接刀！空手者持續觀望；持股者請趁反彈優先減碼降倉。")
        else: # bull_score == 0
            st.error("❄️ **【強勢空頭 - 嚴控風險】**\n\n**戰況：** 四大指標全面破底，系統性風險極高。\n\n**行動：** 提高現金水位，**絕對禁止往下攤平**！耐心等候長期大底出現。")
            
        # --- 新聞顯示區 ---
        st.markdown("---")
        st.subheader("📰 近期相關新聞與情緒判定")
        if news_items_processed:
            for i, news in enumerate(news_items_processed, 1):
                st.markdown(f"**{i}. {news['sentiment']} | [{news['title']}]({news['url']})**")
                st.caption(f"來源: {news['publisher']} | 時間: {news['date']}")
        else:
            st.write("目前找不到最新新聞。")
