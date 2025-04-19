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
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Helper function to update session state
def update_session_state():
    st.session_state['ticker'] = ticker
    st.session_state['period'] = period
    with st.spinner(f"Fetching data for {ticker}..."):
        # Get the data
        st.session_state['stock_data'] = get_stock_data(ticker, period)
        # Record update time
        st.session_state['last_update'] = datetime.datetime.now()

# Update data if fetch button is clicked
if fetch_data:
    update_session_state()

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
        tab1, tab2, tab3, tab4 = st.tabs(["Price Charts", "Moving Averages", "Bollinger Bands", "Data Table"])
        
        with tab1:
            st.subheader("Stock Price Chart")
            # Simple line chart of closing prices
            price_chart_data = pd.DataFrame(stock_data['Close']).rename(columns={'Close': 'Price'})
            st.line_chart(price_chart_data, use_container_width=True)
            
            # Show OHLC data
            st.subheader("Price Details")
            ohlc_data = stock_data[['Open', 'High', 'Low', 'Close']].tail(10)
            st.dataframe(ohlc_data, use_container_width=True)
        
        with tab2:
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
        
        with tab3:
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
            
        with tab4:
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