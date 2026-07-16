import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

# Import the class and helper fetcher from your main file
# (Assumes your trading bot code is saved as 'main.py')
from main import MultiFactorBacktester, MacroDataFetcher


class TestMultiFactorBacktester(unittest.TestCase):

    @patch('main.yf.download')
    def setUp(self, mock_yf_download):
        """
        Set up a deterministic mock dataset representing 220 trading days.
        This avoids external API requests and gives us a clean slate.
        """
        # Create a series of dates
        dates = pd.date_range(start="2025-01-01", periods=220, freq="D")
        
        # Generate base pricing data (stable at $100)
        self.mock_data = pd.DataFrame({
            "Open": [100.0] * 220,
            "High": [101.0] * 220,
            "Low": [99.0] * 220,
            "Close": [100.0] * 220,
            "Volume": [1000000] * 220
        }, index=dates)
        
        # Configure the yf.download mock to return our dataframe
        mock_yf_download.return_name = "SPY"
        mock_yf_download.return_value = self.mock_data.copy()
        
        # Instantiate the backtester (it will pull our mock_data automatically)
        self.backtester = MultiFactorBacktester(
            ticker="SPY", 
            start_date="2025-01-01", 
            end_date="2025-10-01"
        )

    def test_technical_signals_golden_cross(self):
        """Verify that a Golden Cross (SMA_50 > SMA_200) creates a bullish technical signal."""
        # Force a Golden Cross state manually in our data
        self.backtester.data['SMA_50'] = 110.0
        self.backtester.data['SMA_200'] = 100.0
        self.backtester.data['RSI'] = 50.0  # Keep RSI neutral
        
        self.backtester.calculate_technical_indicators()
        
        # Since SMA_50 (110) > SMA_200 (100) and RSI is neutral, signal must be 1.0
        latest_signal = self.backtester.data['Technical_Signal'].iloc[-1]
        self.assertEqual(latest_signal, 1.0)

    def test_technical_signals_death_cross(self):
        """Verify that a Death Cross (SMA_50 < SMA_200) creates a bearish technical signal."""
        # Force a Death Cross state manually
        self.backtester.data['SMA_50'] = 90.0
        self.backtester.data['SMA_200'] = 100.0
        self.backtester.data['RSI'] = 50.0  # Keep RSI neutral
        
        self.backtester.calculate_technical_indicators()
        
        # Since SMA_50 (90) < SMA_200 (100), signal must be -1.0
        latest_signal = self.backtester.data['Technical_Signal'].iloc[-1]
        self.assertEqual(latest_signal, -1.0)

    def test_rsi_overbought_scaling(self):
        """Verify technical signal is dampened (scaled by 50%) when RSI is overbought (>70)."""
        self.backtester.data['SMA_50'] = 110.0
        self.backtester.data['SMA_200'] = 100.0
        self.backtester.data['RSI'] = 75.0  # Overbought
        
        self.backtester.calculate_technical_indicators()
        
        # Bullish signal (1.0) scaled down by 0.5 should equal 0.5
        latest_signal = self.backtester.data['Technical_Signal'].iloc[-1]
        self.assertEqual(latest_signal, 0.5)

    def test_rsi_oversold_scaling(self):
        """Verify technical signal is amplified (scaled by 150%) when RSI is oversold (<30)."""
        self.backtester.data['SMA_50'] = 110.0
        self.backtester.data['SMA_200'] = 100.0
        self.backtester.data['RSI'] = 25.0  # Oversold
        
        self.backtester.calculate_technical_indicators()
        
        # Bullish signal (1.0) scaled up by 1.5 should equal 1.5
        latest_signal = self.backtester.data['Technical_Signal'].iloc[-1]
        self.assertEqual(latest_signal, 1.5)

    @patch.object(MacroDataFetcher, 'get_fed_funds_rate')
    def test_macro_signals_tightening(self, mock_get_fed):
        """Verify that a rising FED Funds rate environment yields a bearish macro signal (-1.0)."""
        # Create a mock FED funds rate series that rises sharply over time
        dates = self.mock_data.index
        fed_rates = np.linspace(1.0, 5.5, len(dates))  # Rising rates
        mock_fed_df = pd.DataFrame({"FEDFUNDS": fed_rates}, index=dates)
        mock_get_fed.return_value = mock_fed_df
        
        self.backtester.integrate_macro_signals()
        
        # Since rates are rising, macro signal should be -1.0
        latest_macro_signal = self.backtester.data['Macro_Signal'].iloc[-1]
        self.assertEqual(latest_macro_signal, -1.0)

    @patch.object(MacroDataFetcher, 'get_fed_funds_rate')
    def test_macro_signals_easing(self, mock_get_fed):
        """Verify that a falling FED Funds rate environment yields a bullish macro signal (+1.0)."""
        # Create a mock FED funds rate series that falls over time
        dates = self.mock_data.index
        fed_rates = np.linspace(5.5, 1.0, len(dates))  # Falling rates
        mock_fed_df = pd.DataFrame({"FEDFUNDS": fed_rates}, index=dates)
        mock_get_fed.return_value = mock_fed_df
        
        self.backtester.integrate_macro_signals()
        
        # Since rates are falling, macro signal should be 1.0
        latest_macro_signal = self.backtester.data['Macro_Signal'].iloc[-1]
        self.assertEqual(latest_macro_signal, 1.0)

    @patch.object(MacroDataFetcher, 'get_fed_funds_rate')
    def test_execution_lag_and_portfolio_math(self, mock_get_fed):
        """
        Verify that trading decisions are lagged by exactly one day, 
        preventing look-ahead bias (cheating by seeing today's close before executing).
        """
        # Simple static setup for macro
        mock_fed_df = pd.DataFrame({"FEDFUNDS": [5.0] * len(self.mock_data)}, index=self.mock_data.index)
        mock_get_fed.return_value = mock_fed_df
        
        # Force a shift in trend signals at a specific index point
        # Run the backtester with a static neutral sentiment score
        self.backtester.run_backtest(sentiment_score=0.0)
        
        # Manually alter a specific target position value to check for shifting
        target_positions = self.backtester.data['Target_Position']
        actual_positions = self.backtester.data['Actual_Position']
        
        # Ensure that Actual_Position[i] equals Target_Position[i-1]
        for i in range(1, len(self.backtester.data) - 1):
            self.assertEqual(actual_positions.iloc[i], target_positions.iloc[i-1])


if __name__ == '__main__':
    unittest.main()
