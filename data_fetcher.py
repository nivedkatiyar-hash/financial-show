import yfinance as yf
import pandas as pd
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime

class DataFetcher:
    """
    Handles the ingestion of Market, Macro, and Sentiment data.
    """

    @staticmethod
    def get_market_data(ticker, period="2y"):
        """Fetches OHLCV data from Yahoo Finance."""
        print(f"[*] Fetching Market Data for {ticker}...")
        df = yf.download(ticker, period=period)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df

    @staticmethod
    def get_fed_funds_rate():
        """Fetches FEDFUNDS data from St. Louis FRED."""
        print("[*] Fetching FED Macro Data...")
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
        try:
            df = pd.read_csv(url)
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df[df['FEDFUNDS'] != '.']
            df['FEDFUNDS'] = df['FEDFUNDS'].astype(float)
            df.set_index('DATE', inplace=True)
            return df
        except Exception:
            return None

    @staticmethod
    def get_sentiment_score():
        """Fetches RSS headlines and calculates a simple sentiment score."""
        print("[*] Fetching News Sentiment...")
        feed_url = "https://www.cnbc.com/id/100003114/device/rss/rss.html"
        try:
            req = urllib.request.Request(feed_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                root = ET.fromstring(response.read())
                
            headlines = [item.find("title").text for item in root.findall(".//item")]
            
            # Simple lexicon scoring
            bullish = ["rally", "growth", "boost", "easing"]
            bearish = ["crash", "recession", "hike", "risk"]
            
            score = 0
            for h in headlines[:10]:
                text = h.lower()
                for b in bullish: score += 1 if b in text else 0
                for r in bearish: score -= 1 if r in text else 0
            
            return np.clip(score / 10, -1.0, 1.0) # Normalize
        except:
            return 0.0 # Neutral if news fails

    @classmethod
    def get_consolidated_data(cls, ticker):
        """
        The main entry point: combines all data into one dataframe.
        """
        market = cls.get_market_data(ticker)
        fed = cls.get_fed_funds_rate()
        sentiment = cls.get_sentiment_score()
        
        # Merge Macro Data
        df = market.join(fed, how='left').ffill().bfill()
        
        # Add sentiment as a constant column for the backtester
        df['Sentiment'] = sentiment
        
        return df

# --- Optional: Test this file alone ---
if __name__ == "__main__":
    data = DataFetcher.get_consolidated_data("SPY")
    print("\n--- Data Preview ---")
    print(data.tail())
