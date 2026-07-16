import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_plot_trend(ticker_symbol, period="2y"):
    """
    Fetches market data, calculates trends using moving averages, 
    and generates an annotated chart.
    """
    print(f"[*] Fetching market data for {ticker_symbol}...")
    
    # 1. Import Market Data
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period=period)
    
    if df.empty:
        print("[!] No data found. Check the ticker symbol.")
        return
        
    # 2. Analyze Trends (Calculate Simple Moving Averages)
    # The 50-day SMA represents the short-term trend
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    # The 200-day SMA represents the long-term macroeconomic trend
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Determine the current trend state based on the most recent day
    current_price = df['Close'].iloc[-1]
    current_sma50 = df['SMA_50'].iloc[-1]
    current_sma200 = df['SMA_200'].iloc[-1]
    
    if current_sma50 > current_sma200:
        trend_status = "BULLISH (Golden Cross Regime)"
    elif current_sma50 < current_sma200:
        trend_status = "BEARISH (Death Cross Regime)"
    else:
        trend_status = "NEUTRAL (Consolidation)"
        
    # 3. Create Chart with Explanations
    plt.figure(figsize=(12, 6))
    
    # Plotting the price and the two averages
    plt.plot(df.index, df['Close'], label='Close Price', alpha=0.4, color='gray')
    plt.plot(df.index, df['SMA_50'], label='50-Day SMA (Short)', color='blue', linewidth=2)
    plt.plot(df.index, df['SMA_200'], label='200-Day SMA (Long)', color='red', linewidth=2)
    
    # Formatting the chart
    plt.title(f"{ticker_symbol} Market Trend Analysis", fontsize=16, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Adding a dynamic text box that explains the analysis to the user
    explanation_text = (
        f"CURRENT ANALYSIS:\n"
        f"Price: ${current_price:.2f}\n"
        f"Trend: {trend_status}\n\n"
        f"Explanation:\n"
        f"When the short-term 50-SMA (blue) is\n"
        f"above the long-term 200-SMA (red),\n"
        f"the asset is in a sustained uptrend.\n"
        f"If blue crosses below red, it signals\n"
        f"a macro downtrend."
    )
    
    # Place text box on the upper left of the chart
    plt.text(0.02, 0.95, explanation_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # You can change "AAPL" to any ticker, like "SPY" for the S&P 500 or "BTC-USD" for Bitcoin
    analyze_and_plot_trend("AAPL", period="2y")
