import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def calculate_rsi(data, window=14):
    """
    Calculates the Relative Strength Index (RSI) for a given pandas Series.
    """
    # Calculate daily price changes
    delta = data.diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Calculate the exponential moving average of gains and losses
    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()
    
    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analyze_and_plot_trend(ticker_symbol, period="2y"):
    """
    Fetches market data, calculates SMAs and RSI, and plots stacked charts.
    """
    print(f"[*] Fetching market data for {ticker_symbol}...")
    
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period=period)
    
    if df.empty:
        print("[!] No data found. Check the ticker symbol.")
        return
        
    # --- Calculate Indicators ---
    # Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Relative Strength Index (14-day standard)
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    
    current_price = df['Close'].iloc[-1]
    current_rsi = df['RSI'].iloc[-1]
    
    # --- Create Stacked Charts ---
    # Create a figure with 2 subplots. The 'gridspec_kw' makes the top chart taller.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Top Chart: Price and Moving Averages
    ax1.plot(df.index, df['Close'], label='Close Price', alpha=0.5, color='gray')
    ax1.plot(df.index, df['SMA_50'], label='50-Day SMA', color='blue', linewidth=1.5)
    ax1.plot(df.index, df['SMA_200'], label='200-Day SMA', color='red', linewidth=1.5)
    
    ax1.set_title(f"{ticker_symbol} Trend and Momentum Analysis", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    
    # Bottom Chart: RSI Oscillator
    ax2.plot(df.index, df['RSI'], label='RSI (14-day)', color='purple', linewidth=1.5)
    
    # Add horizontal lines for overbought (70) and oversold (30) thresholds
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    
    # Fill the areas to highlight overbought/oversold zones visually
    ax2.fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70), facecolor='red', alpha=0.3, interpolate=True)
    ax2.fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30), facecolor='green', alpha=0.3, interpolate=True)
    
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, 100) # RSI is always bound between 0 and 100
    ax2.grid(alpha=0.3)
    
    # Add a text readout of the current RSI state
    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
    ax2.text(0.02, 0.85, f"Current RSI: {current_rsi:.1f} ({rsi_status})", transform=ax2.transAxes, 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_and_plot_trend("AAPL", period="2y")
    analyze_and_plot_trend("AAPL", period="2y")
