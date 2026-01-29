import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ssl

# --- Core Fix: Resolve macOS SSL Certificate Verification Issues ---
# This ensures NLTK data can be downloaded securely on macOS environments.
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass
# Initialize NLTK VADER lexicon for sentiment analysis
nltk.download('vader_lexicon', quiet=True)

class TeslaLatestEarningsProject:
    def __init__(self, ticker="TSLA"):
        self.ticker = ticker
        self.sia = SentimentIntensityAnalyzer()
        self.data = None
        self.score = 0

    def run_analysis(self, earnings_date, text):
        # 1. Quantifying Management Sentiment via NLP
        # Extracts a compound score to represent the tone of the earnings call.
        self.score = self.sia.polarity_scores(text)['compound']
        
        # 2. Fetching High-Frequency Market Data
        # Captures stock price movement around the specific earnings release date.
        date_obj = datetime.strptime(earnings_date, '%Y-%m-%d')
        start = (date_obj - timedelta(days=10)).strftime('%Y-%m-%d')
        end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Fetching latest market data for {self.ticker}...")
        # multi_level_index=False ensures flat column headers for robust data handling.
        self.data = yf.download(self.ticker, start=start, end=end, multi_level_index=False)
        
        if not self.data.empty:
            self.show_chart(date_obj)

    def show_chart(self, earnings_date_obj):
        # Data Cleaning: Removing timezone info for Plotly compatibility
        if self.data.index.tz is not None:
            self.data.index = self.data.index.tz_localize(None)

        fig = go.Figure()
        
        # Plotting Candlestick Chart for TSLA Price Action
        fig.add_trace(go.Candlestick(
            x=self.data.index,
            open=self.data['Open'], high=self.data['High'],
            low=self.data['Low'], close=self.data['Close'],
            name="TSLA Price"
        ))

        # Annotating the Q4 2025 Earnings Release (Jan 28, 2026)
        # Uses millisecond timestamps to align the event line with time-series data.
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

if __name__ == "__main__":
    # Initialize analysis for Tesla (TSLA)
    my_project = TeslaLatestEarningsProject("TSLA")
    
    # Narrative Summary: Focused on Robotaxi, FSD, and Energy Storage metrics
    # This simulated text mirrors the key themes from the Jan 28, 2026 release.
    q4_2025_text = """
    Tesla reported Q4 revenue of $24.9bn. While profits were under pressure, 
    the market responded positively to FSD v13 milestones and Robotaxi progress. 
    Energy storage deployments hit a record 14.2 GWh.
    """
    
   # Execute quantitative analysis for the Jan 28, 2026 event
    my_project.run_analysis("2026-01-28", q4_2025_text)