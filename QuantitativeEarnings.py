import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ssl

# --- 核心修复：解决 Mac SSL 证书问题 ---
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

nltk.download('vader_lexicon', quiet=True)

class TeslaLatestEarningsProject:
    def __init__(self, ticker="TSLA"):
        self.ticker = ticker
        self.sia = SentimentIntensityAnalyzer()
        self.data = None
        self.score = 0

    def run_analysis(self, earnings_date, text):
        # 1. 实时情绪分析
        self.score = self.sia.polarity_scores(text)['compound']
        
        # 2. 获取最新的市场数据 (包含 2026-01-28)
        date_obj = datetime.strptime(earnings_date, '%Y-%m-%d')
        start = (date_obj - timedelta(days=10)).strftime('%Y-%m-%d')
        end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Fetching latest market data for {self.ticker}...")
        # multi_level_index=False 确保 K 线图能正常显示
        self.data = yf.download(self.ticker, start=start, end=end, multi_level_index=False)
        
        if not self.data.empty:
            self.show_chart(date_obj)

    def show_chart(self, earnings_date_obj):
        if self.data.index.tz is not None:
            self.data.index = self.data.index.tz_localize(None)

        fig = go.Figure()
        
        # 绘制特斯拉最新 K 线图
        fig.add_trace(go.Candlestick(
            x=self.data.index,
            open=self.data['Open'], high=self.data['High'],
            low=self.data['Low'], close=self.data['Close'],
            name="TSLA Price"
        ))

        # 标记昨天刚刚发布的 2025 Q4 财报日
        ms_timestamp = earnings_date_obj.timestamp() * 1000
        fig.add_vline(
            x=ms_timestamp, 
            line_dash="dash", 
            line_color="gold", 
            annotation_text=f"Q4 2025 Earnings (Score: {self.score:.2f})",
            annotation_position="top left"
        )

        fig.update_layout(
            title=f"TSLA Q4 2025 Earnings Analysis: Real-time Market Reaction",
            yaxis_title="Stock Price (USD)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )
        
        print(f"Latest TSLA Close Price: ${self.data['Close'].iloc[-1]:.2f}")
        fig.show()

# --- 注意：这部分的缩进必须严格统一 ---
if __name__ == "__main__":
    my_project = TeslaLatestEarningsProject("TSLA")
    
    # 模拟文本：基于 2026 年 1 月 28 日的财报叙事
    q4_2025_text = """
    Tesla reported Q4 revenue of $24.9bn. While profits were under pressure, 
    the market responded positively to FSD v13 milestones and Robotaxi progress. 
    Energy storage deployments hit a record 14.2 GWh.
    """
    
    # 分析日期设为昨天 1 月 28 日
    my_project.run_analysis("2026-01-28", q4_2025_text)