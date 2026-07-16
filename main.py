import os
import logging
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Dict, Any

# ==============================================================================
# 1. CONFIGURATION & LOGGING (Professional Setup)
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantBot")

class SystemConfig:
    """Centralized configuration for the bot."""
    INITIAL_CAPITAL = 100000.0
    FEE_PER_TRADE = 1.00 # Fixed commission
    SLIPPAGE_PCT = 0.0005 # 0.05% price slippage
    DEFAULT_TICKER = "SPY"
    DATE_START = "2020-01-01"
    DATE_END = datetime.datetime.now().strftime("%Y-%m-%d")

# ==============================================================================
# 2. DATA INGESTION ENGINE
# ==============================================================================
class DataEngine:
    """Handles fetching and cleaning data from external sources."""
    @staticmethod
    def fetch_market_data(ticker: str) -> pd.DataFrame:
        logger.info(f"Fetching market data for {ticker}")
        df = yf.download(ticker, start=SystemConfig.DATE_START, end=SystemConfig.DATE_END)
        # Handle multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Processing technical indicators...")
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        return df.dropna()

# ==============================================================================
# 3. STRATEGY ENGINE (Polymorphism)
# ==============================================================================
class StrategyBase:
    """Base class for all strategies."""
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

class TrendFollowing(StrategyBase):
    """Signals: Buy when SMA_50 > SMA_200 (Golden Cross)."""
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
        return df

class MeanReversion(StrategyBase):
    """Signals: Buy when RSI < 30 (Oversold)."""
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Signal'] = np.where(df['RSI'] < 30, 1, 0)
        return df

# ==============================================================================
# 4. BACKTESTER ENGINE
# ==============================================================================
class Backtester:
    """Executes the simulation logic."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = None

    def run(self):
        logger.info("Starting backtest loop...")
        capital = SystemConfig.INITIAL_CAPITAL
        shares = 0
        portfolio_values = []

        for date, row in self.df.iterrows():
            price = row['Close']
            signal = row['Signal']
            
            # Simplified Logic: If signal is 1, go long
            if signal == 1 and shares == 0:
                shares = (capital - SystemConfig.FEE_PER_TRADE) // (price * (1 + SystemConfig.SLIPPAGE_PCT))
                capital -= (shares * price * (1 + SystemConfig.SLIPPAGE_PCT)) + SystemConfig.FEE_PER_TRADE
            
            # Exit if signal is 0 (or some other condition)
            elif signal == 0 and shares > 0:
                capital += (shares * price * (1 - SystemConfig.SLIPPAGE_PCT)) - SystemConfig.FEE_PER_TRADE
                shares = 0
            
            portfolio_values.append(capital + (shares * price))

        self.df['Portfolio_Value'] = portfolio_values
        logger.info("Backtest complete.")

# ==============================================================================
# 5. VISUALIZATION ENGINE
# ==============================================================================
class VisualizationEngine:
    """Generates charts for performance analysis."""
    @staticmethod
    def plot_results(df: pd.DataFrame):
        logger.info("Generating performance charts...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        ax1.plot(df.index, df['Close'], label="Price", color='black', alpha=0.6)
        ax1.set_title("Strategy Performance")
        ax1.legend()
        
        ax2.plot(df.index, df['Portfolio_Value'], label="Portfolio Equity", color='green')
        ax2.set_title("Equity Curve")
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# ==============================================================================
# 6. ORCHESTRATOR
# ==============================================================================
def main():
    logger.info("--- INITIALIZING QUANT SYSTEM ---")
    
    # 1. Fetch & Prepare
    data = DataEngine.fetch_market_data(SystemConfig.DEFAULT_TICKER)
    data = DataEngine.add_indicators(data)
    
    # 2. Select Strategy (Easily toggleable)
    strategy = TrendFollowing() # Change to MeanReversion() to switch logic
    data = strategy.generate_signals(data)
    
    # 3. Backtest
    bot = Backtester(data)
    bot.run()
    
    # 4. Display
    VisualizationEngine.plot_results(bot.df)
    
    logger.info("--- SYSTEM SHUTDOWN ---")

if __name__ == "__main__":
    main()
