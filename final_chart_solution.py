import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

# Set page configuration
st.set_page_config(page_title="Stock Data Visualizer", layout="wide")

# Page title
st.title("Stock Data Visualizer")
st.write("A reliable approach to visualizing stock data with moving averages")

# Sidebar for inputs
with st.sidebar:
    st.header("Stock Selection")
    
    # Stock symbol input
    ticker = st.text_input("Enter Stock Symbol", "AAPL").upper()
    
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
                                index=3)  # Default to 1 Year
    period = period_options[period_label]
    
    # Get data button
    fetch_data = st.button("Fetch Stock Data", use_container_width=True)

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
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Main content area
if fetch_data:
    with st.spinner(f"Fetching data for {ticker}..."):
        # Get the data
        stock_data = get_stock_data(ticker, period)
        
    if stock_data is None or stock_data.empty:
        st.error(f"No data found for {ticker}")
    else:
        # Display basic info
        st.header(f"{ticker} Stock Data")
        
        # Display date range
        start_date = stock_data.index.min().strftime('%Y-%m-%d')
        end_date = stock_data.index.max().strftime('%Y-%m-%d')
        
        # Price info
        current_price = stock_data['Close'].iloc[-1]
        price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
        price_change_pct = (price_change / stock_data['Close'].iloc[-2]) * 100
        
        # Format as green/red based on positive/negative
        price_color = "green" if price_change >= 0 else "red"
        change_symbol = "+" if price_change >= 0 else ""
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", 
                      f"{change_symbol}{price_change:.2f} ({change_symbol}{price_change_pct:.2f}%)",
                      delta_color=price_color)
        
        with col2:
            st.metric("Date Range", f"{start_date} to {end_date}")
            
        with col3:
            min_price = float(stock_data['Low'].min())
            max_price = float(stock_data['High'].max())
            st.metric("Price Range", f"${min_price:.2f} - ${max_price:.2f}")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Price Charts", "Moving Averages", "Data Table"])
        
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
            
            # Select which MAs to display
            ma_options = st.multiselect(
                "Select Moving Averages to Display",
                options=["20-Day MA", "50-Day MA", "100-Day MA", "200-Day MA"],
                default=["20-Day MA", "50-Day MA"]
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