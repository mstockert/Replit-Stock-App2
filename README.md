# Stock Data Visualizer

A comprehensive Streamlit-based stock analysis platform that offers advanced financial insights through interactive visualization and technical analysis tools.

![Stock Data Visualizer](generated-icon.png)

## Features

- **Real-time Stock Data**: Fetch current and historical stock data from Yahoo Finance
- **Multi-stock Comparison**: Compare performance of multiple stocks simultaneously
- **Interactive Charts**: View stock price history with multiple visualization options
- **Technical Indicators**:
  - Multiple Moving Averages (20, 50, 100, 200-day)
  - Bollinger Bands (20-day with 2 standard deviations)
- **Data Analysis**: Export stock data to CSV for further analysis
- **User-friendly Interface**: Clean, organized layout with tabs for different analyses
- **State Persistence**: Maintains application state during interaction using Streamlit's session state
- **Performance Normalization**: Option to normalize stock prices for better comparison of relative performance

## How to Use

### Single Stock Analysis
1. Enter a stock symbol in the sidebar (e.g., AAPL, MSFT, GOOGL)
2. Select a time period (1 month to max)
3. Click "Fetch Stock Data"
4. Navigate between different tabs to explore various analyses:
   - **Price Charts**: View basic price history and recent price details
   - **Moving Averages**: Select different moving averages to display
   - **Bollinger Bands**: Analyze price volatility and potential reversal points
   - **Data Table**: View and download the raw stock data

### Multi-Stock Comparison
1. Enable "Stock Comparison" in the sidebar
2. Enter multiple stock symbols separated by commas (e.g., AAPL,MSFT,GOOGL)
3. Choose whether to normalize prices for better relative performance comparison
4. Click "Compare Stocks"
5. Navigate to the "Comparison" tab to view the multi-stock chart

## Technical Analysis Guides

### Moving Averages
- When price crosses above a moving average: Potential bullish signal
- When price crosses below a moving average: Potential bearish signal
- When shorter-term MA crosses above longer-term MA: Golden Cross (bullish)
- When shorter-term MA crosses below longer-term MA: Death Cross (bearish)
- Multiple MAs pointing in same direction: Strong trend confirmation

### Bollinger Bands
- **Middle Band**: 20-day simple moving average (SMA)
- **Upper Band**: SMA + (2 × 20-day standard deviation)
- **Lower Band**: SMA - (2 × 20-day standard deviation)

**Trading Signals:**
- Price touching the upper band may indicate overbought conditions
- Price touching the lower band may indicate oversold conditions
- Bands narrowing suggest consolidation (low volatility)
- Bands widening suggest increased volatility
- Price breaking out after band contraction often signals a significant move

## Technical Stack

- **Streamlit**: Framework for interactive web application
- **yfinance**: Yahoo Finance data integration
- **pandas**: Data manipulation and analysis
- **matplotlib/plotly**: Visualization libraries

## Development Notes

This application has gone through several iterations to address display issues with charts and state management. The current implementation uses:

- Streamlit's native chart components for reliable display
- Session state to maintain application state
- Keys for all interactive components to ensure state persistence
- Proper error handling for data retrieval and processing

Last updated: April 19, 2025