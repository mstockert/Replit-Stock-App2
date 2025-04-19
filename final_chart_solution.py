import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

# Initialize session state to store our data
if 'stock_data' not in st.session_state:
    st.session_state['stock_data'] = None
if 'ticker' not in st.session_state:
    st.session_state['ticker'] = "AAPL"
if 'period' not in st.session_state:
    st.session_state['period'] = "1y"
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = None
if 'ma_selection' not in st.session_state:
    st.session_state['ma_selection'] = ["20-Day MA", "50-Day MA"]
if 'comparison_mode' not in st.session_state:
    st.session_state['comparison_mode'] = False
if 'comparison_tickers' not in st.session_state:
    st.session_state['comparison_tickers'] = []
if 'comparison_data' not in st.session_state:
    st.session_state['comparison_data'] = {}

# Set page configuration
st.set_page_config(page_title="Stock Data Visualizer", layout="wide")

# Page title
st.title("Stock Data Visualizer")
st.write("A reliable approach to visualizing stock data with moving averages")

# Sidebar for inputs
with st.sidebar:
    st.header("Stock Selection")
    
    # Stock symbol input with key for state persistence
    ticker = st.text_input("Enter Stock Symbol", "AAPL", key="ticker_input").upper()
    
    # Period selection
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max": "max"
    }
    
    period_label = st.selectbox("Select Time Period", 
                                options=list(period_options.keys()),
                                index=3,  # Default to 1 Year
                                key="period_selection")
    period = period_options[period_label]
    
    # Add separator for comparison mode
    st.markdown("---")
    st.header("Stock Comparison")
    
    # Toggle for comparison mode
    comparison_mode = st.checkbox("Enable Stock Comparison", value=st.session_state['comparison_mode'], key="comparison_toggle")
    st.session_state['comparison_mode'] = comparison_mode
    
    if comparison_mode:
        # Input for multiple tickers
        comparison_input = st.text_input(
            "Enter Stock Symbols (comma separated)",
            value=",".join(st.session_state['comparison_tickers']) if st.session_state['comparison_tickers'] else "AAPL,MSFT,GOOGL",
            key="comparison_input"
        )
        
        # Parse input to get list of tickers
        if comparison_input:
            comparison_tickers = [ticker.strip().upper() for ticker in comparison_input.split(",") if ticker.strip()]
            st.session_state['comparison_tickers'] = comparison_tickers
            
        # Option for normalized chart
        normalize = st.checkbox("Normalize Prices", value=True, key="normalize_checkbox")
        
        # Comparison fetch button with key
        compare_stocks = st.button("Compare Stocks", use_container_width=True, key="compare_button")
    else:
        compare_stocks = False
    
    # Get data button with a key to maintain state
    fetch_data = st.button("Fetch Stock Data", use_container_width=True, key="fetch_data_button")

# Function to get stock data
@st.cache_data(ttl=3600)  # Cache for one hour
def get_stock_data(symbol, time_period):
    try:
        data = yf.download(symbol, period=time_period)
        if data.empty:
            return None
            
        # Calculate moving averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA100'] = data['Close'].rolling(window=100).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # Calculate Bollinger Bands (20-day, 2 standard deviations)
        rolling_mean = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = rolling_mean + (rolling_std * 2)
        data['BB_Middle'] = rolling_mean
        data['BB_Lower'] = rolling_mean - (rolling_std * 2)
        
        # Calculate RSI (14-day)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        # MACD Line = 12-day EMA - 26-day EMA
        # Signal Line = 9-day EMA of MACD Line
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD_Line'] = ema12 - ema26
        data['MACD_Signal'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD_Line'] - data['MACD_Signal']
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to get comparison data for multiple stocks
@st.cache_data(ttl=3600)  # Cache for one hour
def get_comparison_data(tickers, time_period):
    try:
        # Download data for all tickers at once
        data = yf.download(tickers, period=time_period)
        
        if data.empty:
            return None
        
        # If we get a single stock, the structure is different
        if len(tickers) == 1:
            # Rename columns to include ticker
            ticker = tickers[0]
            result = {}
            result[ticker] = data
            return result
        
        # Extract close prices for comparison
        result = {}
        for ticker in tickers:
            if ticker in data['Close'].columns:
                ticker_data = data.xs(ticker, axis=1, level=1, drop_level=False)
                # Create a new DataFrame with proper structure
                ticker_df = pd.DataFrame()
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if (col, ticker) in ticker_data:
                        ticker_df[col] = ticker_data[(col, ticker)]
                result[ticker] = ticker_df
        
        return result
    except Exception as e:
        st.error(f"Error fetching comparison data: {str(e)}")
        return None

# Function to normalize price data for better comparison
def normalize_data(data_dict):
    normalized = {}
    for ticker, data in data_dict.items():
        if not data.empty:
            # Create a copy to avoid changing the original data
            normalized[ticker] = data.copy()
            # Normalize to the first day's price
            first_price = data['Close'].iloc[0]
            if first_price > 0:  # Avoid division by zero
                normalized[ticker]['Close'] = data['Close'] / first_price * 100
    return normalized

# Helper function to update session state
def update_session_state():
    st.session_state['ticker'] = ticker
    st.session_state['period'] = period
    with st.spinner(f"Fetching data for {ticker}..."):
        # Get the data
        st.session_state['stock_data'] = get_stock_data(ticker, period)
        # Record update time
        st.session_state['last_update'] = datetime.datetime.now()

# Helper function to fetch comparison data
def update_comparison_data():
    # Get comparison tickers from session state
    tickers = st.session_state['comparison_tickers']
    if not tickers:
        st.error("Please enter at least one stock symbol for comparison")
        return
    
    with st.spinner(f"Fetching data for {', '.join(tickers)}..."):
        # Get the data
        comparison_data = get_comparison_data(tickers, period)
        
        if comparison_data:
            st.session_state['comparison_data'] = comparison_data
            st.session_state['last_comparison_update'] = datetime.datetime.now()
        else:
            st.error("Failed to fetch comparison data")

# Update data if fetch button is clicked
if fetch_data:
    update_session_state()

# Update comparison data if compare button is clicked
if st.session_state['comparison_mode'] and 'compare_stocks' in locals() and compare_stocks:
    update_comparison_data()

# Main content area
if st.session_state['stock_data'] is not None:
    # Use data from session state
    stock_data = st.session_state['stock_data']
    ticker = st.session_state['ticker']
    
    if stock_data.empty:
        st.error(f"No data found for {ticker}")
    else:
        # Display basic info
        st.header(f"{ticker} Stock Data")
        
        # Show data update time
        if st.session_state['last_update']:
            st.caption(f"Last updated: {st.session_state['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display date range
        start_date = stock_data.index.min().strftime('%Y-%m-%d')
        end_date = stock_data.index.max().strftime('%Y-%m-%d')
        
        # Price info - handle Series correctly using iloc[0] to avoid warnings
        current_price = float(stock_data['Close'].iloc[-1].iloc[0]) if isinstance(stock_data['Close'].iloc[-1], pd.Series) else float(stock_data['Close'].iloc[-1])
        previous_price = float(stock_data['Close'].iloc[-2].iloc[0]) if isinstance(stock_data['Close'].iloc[-2], pd.Series) else float(stock_data['Close'].iloc[-2])
        price_change = current_price - previous_price
        price_change_pct = (price_change / previous_price) * 100
        
        # Format delta using Streamlit's expected values
        change_symbol = "+" if price_change >= 0 else ""
        # For delta_color, use 'normal' (green for positive, red for negative)
        # or 'inverse' (red for positive, green for negative)
        delta_color = "normal"  # Use Streamlit's built-in coloring system (positive=green, negative=red)
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", 
                      f"{change_symbol}{price_change:.2f} ({change_symbol}{price_change_pct:.2f}%)",
                      delta_color=delta_color)
        
        with col2:
            st.metric("Date Range", f"{start_date} to {end_date}")
            
        with col3:
            # Use iloc[0] as recommended in warning messages
            min_price = float(stock_data['Low'].min().iloc[0]) if isinstance(stock_data['Low'].min(), pd.Series) else float(stock_data['Low'].min())
            max_price = float(stock_data['High'].max().iloc[0]) if isinstance(stock_data['High'].max(), pd.Series) else float(stock_data['High'].max())
            st.metric("Price Range", f"${min_price:.2f} - ${max_price:.2f}")
        
        # Tabs for different views
        tab_list = ["Price Charts", "Moving Averages", "Bollinger Bands", "RSI", "MACD", "Data Table"]
        
        # Add comparison tab if in comparison mode
        if st.session_state['comparison_mode'] and st.session_state['comparison_data']:
            tab_list.insert(0, "Comparison")
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_list)
        else:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_list)
        
        # Handle the comparison tab if it exists
        tab_index = 0  # Keep track of which tab we're on
        
        if st.session_state['comparison_mode'] and st.session_state['comparison_data']:
            with tab1:
                st.subheader("Stock Comparison")
                
                if st.session_state['last_comparison_update']:
                    st.caption(f"Last updated: {st.session_state['last_comparison_update'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Get normalize value from the sidebar (or default to True if not set)
                normalize = locals().get('normalize', True) 
                
                # Prepare data for chart
                comparison_chart_data = pd.DataFrame()
                
                # Process the comparison data
                comp_data = st.session_state['comparison_data']
                
                # Use either normalized or raw data based on selection
                if normalize:
                    processed_data = normalize_data(comp_data)
                    chart_title = "Normalized Stock Comparison (First Day = 100)"
                else:
                    processed_data = comp_data
                    chart_title = "Stock Price Comparison"
                
                # Extract close prices for all tickers
                for ticker, data in processed_data.items():
                    if not data.empty:
                        comparison_chart_data[ticker] = data['Close']
                
                # Display the comparison chart
                st.subheader(chart_title)
                if not comparison_chart_data.empty:
                    st.line_chart(comparison_chart_data, use_container_width=True)
                else:
                    st.error("No comparison data to display")
                
                # Information about the comparison
                st.write("""
                ### Comparison Analysis
                
                This chart allows you to compare the performance of multiple stocks over time.
                
                **When normalized:**
                - All stocks start at a base value of 100
                - Chart shows percentage changes relative to starting point
                - Better for comparing performance regardless of actual price
                
                **When not normalized:**
                - Shows actual stock prices
                - Better for comparing actual price levels and movements
                """)
                
                # Increment tab index
                tab_index += 1
            
            # Next tab becomes the price chart tab
            with tab2:
                st.subheader("Stock Price Chart")
                # Simple line chart of closing prices
                price_chart_data = pd.DataFrame(stock_data['Close']).rename(columns={'Close': 'Price'})
                st.line_chart(price_chart_data, use_container_width=True)
                
                # Show OHLC data
                st.subheader("Price Details")
                ohlc_data = stock_data[['Open', 'High', 'Low', 'Close']].tail(10)
                st.dataframe(ohlc_data, use_container_width=True)
                
                # Increment tab index
                tab_index += 1
        else:
            # If no comparison, the first tab is the price chart
            with tab1:
                st.subheader("Stock Price Chart")
                # Simple line chart of closing prices
                price_chart_data = pd.DataFrame(stock_data['Close']).rename(columns={'Close': 'Price'})
                st.line_chart(price_chart_data, use_container_width=True)
                
                # Show OHLC data
                st.subheader("Price Details")
                ohlc_data = stock_data[['Open', 'High', 'Low', 'Close']].tail(10)
                st.dataframe(ohlc_data, use_container_width=True)
                
                # Increment tab index
                tab_index += 1
        
        # Moving Averages tab - this will be tab2 or tab3 depending on whether comparison is active
        if st.session_state['comparison_mode'] and st.session_state['comparison_data']:
            current_tab = tab3  # If comparison is active, MA tab is the 3rd tab
        else:
            current_tab = tab2  # Otherwise, it's the 2nd tab
            
        with current_tab:
            st.subheader("Moving Average Analysis")
            
            # Select which MAs to display - use key to maintain state
            # Adding a key prevents the selector from reverting to default on interaction
            ma_options = st.multiselect(
                "Select Moving Averages to Display",
                options=["20-Day MA", "50-Day MA", "100-Day MA", "200-Day MA"],
                default=["20-Day MA", "50-Day MA"],
                key="ma_selection"  # Adding a key helps Streamlit maintain state
            )
            
            # Create dataframe for display
            ma_data = pd.DataFrame(index=stock_data.index)
            ma_data['Price'] = stock_data['Close']
            
            # Map selections to columns
            ma_mapping = {
                "20-Day MA": "MA20", 
                "50-Day MA": "MA50",
                "100-Day MA": "MA100",
                "200-Day MA": "MA200"
            }
            
            # Add selected MAs to display dataframe
            for ma in ma_options:
                ma_data[ma] = stock_data[ma_mapping[ma]]
            
            # Always show a chart, even if no MAs are selected
            # The price line will still be visible
            st.line_chart(ma_data, use_container_width=True)
            
            # Additional info about moving averages
            st.write("""
            ### Moving Average Interpretation
            
            - When price crosses above a moving average: Potential bullish signal
            - When price crosses below a moving average: Potential bearish signal
            - When shorter-term MA crosses above longer-term MA: Golden Cross (bullish)
            - When shorter-term MA crosses below longer-term MA: Death Cross (bearish)
            - Multiple MAs pointing in same direction: Strong trend confirmation
            """)
        
        # Bollinger Bands tab - this will be tab3 or tab4 depending on whether comparison is active
        if st.session_state['comparison_mode'] and st.session_state['comparison_data']:
            current_tab = tab4  # If comparison is active, BB tab is the 4th tab
        else:
            current_tab = tab3  # Otherwise, it's the 3rd tab
            
        with current_tab:
            st.subheader("Bollinger Bands Analysis")
            
            # Create dataframe for Bollinger Bands display
            bb_data = pd.DataFrame(index=stock_data.index)
            bb_data['Price'] = stock_data['Close']
            bb_data['Upper Band'] = stock_data['BB_Upper']
            bb_data['Middle Band'] = stock_data['BB_Middle']
            bb_data['Lower Band'] = stock_data['BB_Lower']
            
            # Display the chart
            st.line_chart(bb_data, use_container_width=True)
            
            # Explanation of Bollinger Bands
            st.write("""
            ### Bollinger Bands Interpretation
            
            Bollinger Bands consist of three lines:
            - **Middle Band**: 20-day simple moving average (SMA)
            - **Upper Band**: SMA + (2 × 20-day standard deviation)
            - **Lower Band**: SMA - (2 × 20-day standard deviation)
            
            **Trading Signals:**
            - Price touching the upper band may indicate overbought conditions
            - Price touching the lower band may indicate oversold conditions
            - Bands narrowing suggest consolidation (low volatility)
            - Bands widening suggest increased volatility
            - Price breaking out after band contraction often signals a significant move
            """)
        
        # RSI tab
        if st.session_state['comparison_mode'] and st.session_state['comparison_data']:
            current_tab = tab5  # If comparison is active, RSI tab is the 5th tab
        else:
            current_tab = tab4  # Otherwise, it's the 4th tab
            
        with current_tab:
            st.subheader("Relative Strength Index (RSI)")
            
            # Create dataframe for RSI display
            rsi_data = pd.DataFrame(index=stock_data.index)
            rsi_data['RSI'] = stock_data['RSI']
            
            # Display the RSI chart
            st.line_chart(rsi_data, use_container_width=True)
            
            # Add reference lines for overbought and oversold levels
            st.write("**Reference Levels:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**70**: Overbought")
            with col2:
                st.markdown("**50**: Neutral")
            with col3:
                st.markdown("**30**: Oversold")
            
            # Explanation of RSI
            st.write("""
            ### RSI Interpretation
            
            The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
            
            **Key Features:**
            - RSI ranges from 0 to 100
            - Typically, RSI above 70 indicates **overbought** conditions (potential sell signal)
            - RSI below 30 indicates **oversold** conditions (potential buy signal)
            - The centerline (50) can indicate trend direction
                * Above 50: Generally bullish conditions
                * Below 50: Generally bearish conditions
            
            **Trading Signals:**
            - Divergence between RSI and price can signal potential reversals
            - RSI failure swings (when RSI fails to make a new high/low while price does) can be powerful signals
            - Staying in extreme territories during strong trends is common and not always a reversal signal
            """)
        
        # MACD tab
        if st.session_state['comparison_mode'] and st.session_state['comparison_data']:
            current_tab = tab6  # If comparison is active, MACD tab is the 6th tab
        else:
            current_tab = tab5  # Otherwise, it's the 5th tab
            
        with current_tab:
            st.subheader("Moving Average Convergence Divergence (MACD)")
            
            # Create dataframe for MACD display
            macd_data = pd.DataFrame(index=stock_data.index)
            macd_data['MACD Line'] = stock_data['MACD_Line']
            macd_data['Signal Line'] = stock_data['MACD_Signal']
            
            # Display the MACD lines chart
            st.subheader("MACD and Signal Lines")
            st.line_chart(macd_data, use_container_width=True)
            
            # Create a separate chart for the histogram
            st.subheader("MACD Histogram")
            hist_data = pd.DataFrame(index=stock_data.index)
            hist_data['Histogram'] = stock_data['MACD_Histogram']
            st.bar_chart(hist_data, use_container_width=True)
            
            # Explanation of MACD
            st.write("""
            ### MACD Interpretation
            
            The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship 
            between two moving averages of a security's price.
            
            **Components:**
            - **MACD Line**: 12-day EMA - 26-day EMA
            - **Signal Line**: 9-day EMA of the MACD Line
            - **Histogram**: MACD Line - Signal Line
            
            **Trading Signals:**
            - **Crossovers**: When MACD crosses above the signal line, it's a bullish signal; when it crosses below, it's bearish
            - **Divergence**: When price makes a new high/low but MACD doesn't, it can signal a potential reversal
            - **Histogram**: When the histogram gets smaller, momentum is slowing; increasing histogram indicates increasing momentum
            - **Zero Line Crossover**: MACD crossing above zero is bullish; crossing below is bearish
            """)
        
        # Data Table tab - now the last tab
        if st.session_state['comparison_mode'] and st.session_state['comparison_data']:
            current_tab = tab7  # If comparison is active, Data tab is the 7th tab
        else:
            current_tab = tab6  # Otherwise, it's the 6th tab
            
        with current_tab:
            st.subheader("Historical Data")
            st.dataframe(stock_data, use_container_width=True)
            
            # Download button
            csv = stock_data.to_csv().encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{ticker}_data.csv",
                mime="text/csv"
            )
else:
    # Default instructions when no data is loaded
    st.info("Enter a stock symbol and click 'Fetch Stock Data' to begin analysis.")
    
    # Show sample stocks
    st.write("""
    ### Sample Stocks to Try
    - AAPL (Apple)
    - MSFT (Microsoft)
    - GOOGL (Google)
    - AMZN (Amazon)
    - TSLA (Tesla)
    - JPM (JPMorgan Chase)
    - V (Visa)
    - JNJ (Johnson & Johnson)
    """)
    
    # When the app was last updated
    st.write(f"App last updated: {datetime.date.today().strftime('%B %d, %Y')}")