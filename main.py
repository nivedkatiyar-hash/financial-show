"""
Quantitative Multi-Factor Trading System
Integrates Technical Indicators, Federal Reserve (FRED) Macro Data, and RSS News Sentiment.
"""

import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ==============================================================================
# 1. ALTERNATIVE DATA FETCHERS (FED & NEWS)
# ==============================================================================

class MacroDataFetcher:
    """Handles programmatic downloading of economic indicators from St. Louis FRED."""
    
    @staticmethod
    def get_fed_funds_rate() -> pd.DataFrame:
        """
        Fetches the Federal Funds Effective Rate (FEDFUNDS) from FRED.
        Uses public CSV export URLs to bypass proprietary API key requirements.
        """
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
        try:
            print("[*] Downloading Federal Reserve Macro Data (FEDFUNDS)...")
            df = pd.read_csv(url)
            df['DATE'] = pd.to_datetime(df['DATE'])
            # Clean FRED missing-value placeholders (usually '.')
            df = df[df['FEDFUNDS'] != '.']
            df['FEDFUNDS'] = df['FEDFUNDS'].astype(float)
            df.set_index('DATE', inplace=True)
            return df
        except Exception as e:
            print(f"[!] FRED download failed: {e}. Utilizing simulated macro data fallback.")
            # Fallback synthetic historical rate cycle (for offline testing)
            dates = pd.date_range(start="2018-01-01", end=datetime.now(), freq="ME")
            rates = []
            for d in dates:
                if d < pd.Timestamp("2022-03-01"):
                    rates.append(0.10)  # Near-zero era
                elif d < pd.Timestamp("2023-07-01"):
                    rates.append(4.50)  # Hiking cycle
                else:
                    rates.append(5.33)  # Peak high-rate plateau
            return pd.DataFrame({"FEDFUNDS": rates}, index=dates)


class LiveNewsSentiment:
    """Parses public RSS feeds and applies lexicon-based sentiment analysis."""
    
    # Simple financial lexicon mapping
    BULLISH_KEYWORDS = ["rally", "growth", "bullish", "earnings", "upbeat", "boost", "surge", "easing", "cut", "support"]
    BEARISH_KEYWORDS = ["crash", "recession", "inflation", "bearish", "hike", "slowdown", "risk", "drop", "plunge", "deficit"]

    @staticmethod
    def fetch_rss_headlines() -> list:
        """Fetches latest financial headlines from CNBC's live stock feed."""
        feed_url = "https://www.cnbc.com/id/100003114/device/rss/rss.html"
        try:
            print("[*] Fetching live financial headlines from CNBC RSS...")
            req = urllib.request.Request(
                feed_url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()
            
            root = ET.fromstring(xml_data)
            headlines = []
            for item in root.findall(".//item"):
                title = item.find("title")
                if title is not None and title.text:
                    headlines.append(title.text)
            return headlines
        except Exception as e:
            print(f"[!] RSS fetch failed ({e}). Falling back to static mock headlines.")
            return [
                "Markets hold steady ahead of crucial Fed announcement",
                "Inflation fears creep back as regional commodity prices rise",
                "Tech giants rally as cloud demand reaches new record highs"
            ]

    @classmethod
    def calculate_sentiment(cls) -> float:
        """
        Analyzes live news and computes an aggregate sentiment factor score between [-1, 1].
        """
        headlines = cls.fetch_rss_headlines()
        if not headlines:
            return 0.0
        
        total_score = 0.0
        print(f"\n--- LIVE FINANCIAL SENTIMENT MONITOR ---")
        for h in headlines[:8]:  # Analyze top 8 recent headlines
            score = 0.0
            words = h.lower().split()
            for word in words:
                if any(bk in word for bk in cls.BULLISH_KEYWORDS):
                    score += 0.25
                if any(rk in word for rk in cls.BEARISH_KEYWORDS):
                    score -= 0.25
            
            # Normalize single headline score between -1.0 and 1.0
            score = np.clip(score, -1.0, 1.0)
            print(f"  [Score: {score:+.2f}] | {h}")
            total_score += score
            
        avg_sentiment = np.clip(total_score / min(len(headlines), 8), -1.0, 1.0)
        print(f"--- Composite News Sentiment: {avg_sentiment:+.2f} ---\n")
        return avg_sentiment


# ==============================================================================
# 2. QUANT STRATEGY ENGINE
# ==============================================================================

class MultiFactorBacktester:
    """
    Backtests a technical-macro-sentiment hybrid strategy on a selected stock.
    """
    def __init__(self, ticker: str, start_date: str, end_date: str, initial_capital: float = 100000.0):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # Load asset data
        print(f"[*] Downloading market historical pricing for {self.ticker}...")
        self.data = yf.download(ticker, start=start_date, end=end_date)
        if isinstance(self.data.columns, pd.MultiIndex):
            # Flatten multi-index columns if returned by yfinance
            self.data.columns = [col[0] for col in self.data.columns]
            
    def calculate_technical_indicators(self):
        """Calculates trend (SMA) and momentum (RSI) metrics."""
        df = self.data
        # Moving Averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Raw technical signal: +1 if short SMA > long SMA, -1 if below
        df['Technical_Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1.0, -1.0)
        # Moderate technical signal exposure if RSI signals overbought/oversold levels
        df.loc[df['RSI'] > 70, 'Technical_Signal'] *= 0.5
        df.loc[df['RSI'] < 30, 'Technical_Signal'] *= 1.5 # lean aggressively into oversold conditions
        
    def integrate_macro_signals(self):
        """Merges historical FED funds rate trend data as a risk factor."""
        fed_df = MacroDataFetcher.get_fed_funds_rate()
        
        # Forward-fill macro data to match market trading days
        self.data = self.data.join(fed_df, how='left')
        self.data['FEDFUNDS'] = self.data['FEDFUNDS'].ffill().bfill()
        
        # Determine macro trajectory: rate rate change over prior 3 monthly observations (approx 63 trading days)
        self.data['FED_Change_3M'] = self.data['FEDFUNDS'].diff(63)
        
        # Easing cycle (rates decreasing) -> Bullish macro regime (+1.0)
        # Tightening cycle (rates increasing) -> Bearish macro regime (-1.0)
        self.data['Macro_Signal'] = 0.0
        self.data.loc[self.data['FED_Change_3M'] > 0.1, 'Macro_Signal'] = -1.0
        self.data.loc[self.data['FED_Change_3M'] < -0.1, 'Macro_Signal'] = 1.0
        
    def run_backtest(self, sentiment_score: float, w_tech: float = 0.5, w_macro: float = 0.3, w_sent: float = 0.2):
        """
        Executes the backtest using a linear weighting of the computed factors.
        """
        self.calculate_technical_indicators()
        self.integrate_macro_signals()
        
        df = self.data
        df['Sentiment_Signal'] = sentiment_score
        
        # Calculate Blended Multi-Factor Score
        df['Blended_Score'] = (
            (w_tech * df['Technical_Signal']) +
            (w_macro * df['Macro_Signal']) +
            (w_sent * df['Sentiment_Signal'])
        )
        
        # Dynamic Target Positions: Long when aggregate factors are positive, flat otherwise
        df['Target_Position'] = np.where(df['Blended_Score'] > 0.1, 1.0, 0.0)
        
        # Daily execution with a 1-day lag to simulate realistic execution next morning
        df['Actual_Position'] = df['Target_Position'].shift(1).fillna(0.0)
        
        # Calculate Returns
        df['Market_Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Market_Returns'] * df['Actual_Position']
        
        # Equity curves
        df['Market_Equity'] = self.initial_capital * (1 + df['Market_Returns']).cumprod().fillna(1.0)
        df['Strategy_Equity'] = self.initial_capital * (1 + df['Strategy_Returns']).cumprod().fillna(1.0)
        
        self.generate_performance_metrics()
        self.plot_equity_curve()

    def generate_performance_metrics(self):
        """Generates statistical performance data on the completed strategy run."""
        df = self.data
        days = len(df)
        years = days / 252.0
        
        final_equity = df['Strategy_Equity'].iloc[-1]
        cagr = ((final_equity / self.initial_capital) ** (1 / years)) - 1
        
        # Max Drawdown calculation
        peak = df['Strategy_Equity'].cummax()
        drawdown = (df['Strategy_Equity'] - peak) / peak
        max_dd = drawdown.min()
        
        # Sharpe Ratio calculation
        risk_free_rate = 0.045 # Adjusted to represent 2026 baseline cash yields
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = df['Strategy_Returns'] - daily_rf
        sharpe = (excess_returns.mean() / (excess_returns.std() + 1e-9)) * np.sqrt(252)
        
        print("\n" + "="*50)
        print(f" MULTI-FACTOR STRATEGY REPORT: {self.ticker}")
        print("="*50)
        print(f"Period:             {self.start_date} to {self.end_date}")
        print(f"Initial Portfolio:  ${self.initial_capital:,.2f}")
        print(f"Ending Portfolio:   ${final_equity:,.2f}")
        print(f"CAGR (Annualized):  {cagr * 100:.2f}%")
        print(f"Sharpe Ratio:       {sharpe:.2f}")
        print(f"Max Drawdown:       {max_dd * 100:.2f}%")
        print(f"Benchmark Return:   {((df['Market_Equity'].iloc[-1] / self.initial_capital) - 1) * 100:.2f}%")
        print("="*50)

    def plot_equity_curve(self):
        """Visualizes trade outcomes against base asset benchmark performance."""
        df = self.data
        plt.figure(figsize=(12, 6))
        
        plt.plot(df.index, df['Market_Equity'], label=f"Benchmark (Buy & Hold {self.ticker})", color='grey', alpha=0.6, linestyle='--')
        plt.plot(df.index, df['Strategy_Equity'], label="Hybrid Multi-Factor Strategy", color='forestgreen', linewidth=2)
        
        # Secondary axis representing FED Rate shifts
        ax2 = plt.twinx()
        ax2.plot(df.index, df['FEDFUNDS'], label="Fed Funds Target Rate (FRED)", color='blue', alpha=0.3, linewidth=1.5)
        ax2.set_ylabel('Federal Funds Rate (%)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        plt.title(f"Multi-Factor Quant Backtest Engine: {self.ticker}", fontsize=14, fontweight='bold')
        plt.xlabel("Timeline")
        
        # Combine legends from both axes
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.show()


# ==============================================================================
# 3. SYSTEM ORCHESTRATOR
# ==============================================================================

if __name__ == "__main__":
    # Get Live News sentiment score first
    sentiment = LiveNewsSentiment.calculate_sentiment()
    
    # Instantiate engine (Backtesting SPY from 2020 through recent periods)
    backtester = MultiFactorBacktester(
        ticker="SPY", 
        start_date="2020-01-01", 
        end_date="2026-06-30",
        initial_capital=100000.0
    )
    
    # Run strategy with custom factor weights:
    # 50% Technical Indicators, 30% FED Policy Direction, 20% News Sentiment
    backtester.run_backtest(
        sentiment_score=sentiment,
        w_tech=0.50,
        w_macro=0.30,
        w_sent=0.20
    )
