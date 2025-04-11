import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import os

# Import database modules
import database as db
from db_ui import display_watchlist_manager, display_recent_searches, save_search_to_history, cache_stock_data

# Initialize session state for persistent UI state across reruns
if 'ticker' not in st.session_state:
    st.session_state['ticker'] = "AAPL"
    
if 'analysis_type' not in st.session_state:
    st.session_state['analysis_type'] = "Single Stock Analysis"
    
if 'compare_symbols' not in st.session_state:
    st.session_state['compare_symbols'] = "AAPL,MSFT,GOOG"

# Set up page configuration
st.set_page_config(
    page_title="Stock Data Visualization",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("📈 Stock Data Visualization Tool")
st.markdown("""
This app retrieves stock data from Yahoo Finance and visualizes it with interactive charts.
Enter one or more stock symbols to get started!
""")

# Function to get stock data
def get_stock_data(ticker_symbol, period='1y'):
    """Fetch stock data using yfinance"""
    try:
        # Get stock data
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period=period)
        
        # Check if data is empty
        if hist.empty:
            st.error(f"No data found for ticker {ticker_symbol}. Please check the symbol and try again.")
            return None, None
        
        # Get company info
        info = stock.info
        return hist, info
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None
        
# Function to get data for multiple stocks
def get_multiple_stocks_data(ticker_symbols, period='1y'):
    """Fetch data for multiple stock symbols"""
    all_data = {}
    all_info = {}
    
    # Use a spinner to show progress
    with st.spinner(f'Fetching data for {", ".join(ticker_symbols)}...'):
        for ticker in ticker_symbols:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    all_data[ticker] = hist
                    all_info[ticker] = stock.info
                else:
                    st.warning(f"No data found for ticker {ticker}. Skipping.")
            except Exception as e:
                st.warning(f"Error fetching data for {ticker}: {e}")
    
    if not all_data:
        st.error("Could not retrieve data for any of the provided symbols.")
        return None, None
    
    return all_data, all_info

# Function to get key financial metrics
def get_financial_metrics(info):
    """Extract key financial metrics from stock info"""
    metrics = {}
    
    try:
        # Basic info
        metrics['Company Name'] = info.get('longName', 'N/A')
        metrics['Sector'] = info.get('sector', 'N/A')
        metrics['Industry'] = info.get('industry', 'N/A')
        
        # Price metrics
        metrics['Current Price'] = info.get('currentPrice', 'N/A')
        metrics['Previous Close'] = info.get('previousClose', 'N/A')
        metrics['Open'] = info.get('open', 'N/A')
        metrics['Day Low'] = info.get('dayLow', 'N/A')
        metrics['Day High'] = info.get('dayHigh', 'N/A')
        metrics['52 Week Low'] = info.get('fiftyTwoWeekLow', 'N/A')
        metrics['52 Week High'] = info.get('fiftyTwoWeekHigh', 'N/A')
        
        # Volume metrics
        metrics['Volume'] = info.get('volume', 'N/A')
        metrics['Avg Volume'] = info.get('averageVolume', 'N/A')
        
        # Valuation metrics
        metrics['Market Cap'] = info.get('marketCap', 'N/A')
        metrics['P/E Ratio'] = info.get('trailingPE', 'N/A')
        metrics['EPS'] = info.get('trailingEps', 'N/A')
        metrics['Forward P/E'] = info.get('forwardPE', 'N/A')
        metrics['Dividend Yield'] = info.get('dividendYield', 'N/A')
        if metrics['Dividend Yield'] != 'N/A' and metrics['Dividend Yield'] is not None:
            metrics['Dividend Yield'] = f"{metrics['Dividend Yield'] * 100:.2f}%"
            
        # Additional metrics
        metrics['Beta'] = info.get('beta', 'N/A')
        metrics['Target Price'] = info.get('targetMeanPrice', 'N/A')
        
    except Exception as e:
        st.warning(f"Could not retrieve all metrics: {e}")
    
    return metrics

# Function to create price chart
def create_price_chart(data, company_name, time_period):
    """Create an interactive price chart using plotly"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Add volume as bar chart on secondary y-axis
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        yaxis='y2',
        marker=dict(color='rgba(0, 0, 255, 0.3)')
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{company_name} Stock Price ({time_period})',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        xaxis_rangeslider_visible=False,
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Function to create comparison chart for multiple stocks
def create_comparison_chart(stock_data_dict, time_period, normalize=False):
    """Create a chart comparing multiple stocks over time"""
    fig = go.Figure()
    
    # Add each stock's closing price
    for ticker, data in stock_data_dict.items():
        # Get the closing prices
        close_prices = data['Close']
        
        # Normalize if requested (start from 100)
        if normalize:
            close_prices = (close_prices / close_prices.iloc[0]) * 100
        
        # Add line to chart
        fig.add_trace(go.Scatter(
            x=data.index,
            y=close_prices,
            mode='lines',
            name=f'{ticker}',
            line=dict(width=2)
        ))
    
    # Update layout
    title = "Normalized Stock Price Comparison (Base = 100)" if normalize else "Stock Price Comparison"
    y_axis_title = "Normalized Price (Base = 100)" if normalize else "Price ($)"
    
    fig.update_layout(
        title=f'{title} ({time_period})',
        xaxis_title='Date',
        yaxis_title=y_axis_title,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    return fig

# Function to format table data for display
def format_table_data(hist_data):
    """Format historical data for display in table"""
    # Reset index to make Date a column
    df = hist_data.reset_index()
    
    # Format Date column
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Round numeric columns to 2 decimal places
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    for col in numeric_cols:
        if col in df.columns:
            if col == 'Volume':
                df[col] = df[col].astype(int)
            else:
                df[col] = df[col].round(2)
    
    return df

# Create the sidebar for inputs
st.sidebar.header("Stock Parameters")

# Analysis type selection
analysis_type = st.sidebar.radio(
    "Select Analysis Type:",
    ["Single Stock Analysis", "Stock Comparison", "Watchlists"]
)

# Update session state
st.session_state['analysis_type'] = analysis_type

# Define time periods (used across different modes)
time_periods = {
    "1 Week": "1wk", 
    "1 Month": "1mo", 
    "3 Months": "3mo", 
    "6 Months": "6mo", 
    "1 Year": "1y", 
    "2 Years": "2y",
    "5 Years": "5y",
    "Maximum": "max"
}

if analysis_type == "Single Stock Analysis":
    # Single stock analysis
    if 'ticker' in st.session_state:
        default_ticker = st.session_state['ticker']
    else:
        default_ticker = "AAPL"
        
    ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOG):", default_ticker).upper()
    st.session_state['ticker'] = ticker
    
    # Time period selection
    selected_period = st.sidebar.selectbox("Select Time Period", list(time_periods.keys()), index=4)
    period = time_periods[selected_period]
    
    # Submit button
    submit_button = st.sidebar.button("Get Stock Data")
    
    # Show recent searches
    st.sidebar.subheader("Recent Searches")
    recent_searches = db.get_search_history(limit=5)
    
    if recent_searches:
        for search in recent_searches:
            if search['type'] == 'single':
                if st.sidebar.button(f"📊 {search['query']}", key=f"recent_{search['id']}"):
                    ticker = search['query']
                    st.session_state['ticker'] = ticker
                    submit_button = True
    
    compare_stocks = False
    tickers = []
    normalize_prices = False  # Not used in single mode

elif analysis_type == "Stock Comparison":
    # Stock comparison
    # Stock symbol input as a text field
    if 'compare_symbols' in st.session_state:
        default_symbols = st.session_state['compare_symbols']
    else:
        default_symbols = "AAPL,MSFT,GOOG"
        
    tickers_input = st.sidebar.text_input(
        "Enter Stock Symbols (comma-separated, e.g., AAPL,MSFT,GOOG):", 
        default_symbols
    ).upper()
    
    st.session_state['compare_symbols'] = tickers_input
    
    # Convert input to list and clean
    tickers = [ticker.strip() for ticker in tickers_input.split(",") if ticker.strip()]
    
    # Time period selection
    selected_period = st.sidebar.selectbox("Select Time Period", list(time_periods.keys()), index=4)
    period = time_periods[selected_period]
    
    # Normalize prices option
    normalize_prices = st.sidebar.checkbox("Normalize Prices (Base = 100)", value=True,
                                         help="Normalize all stock prices to start at 100 for easier comparison")
    
    # Submit button
    submit_button = st.sidebar.button("Compare Stocks")
    
    # Show recent comparison searches
    st.sidebar.subheader("Recent Comparisons")
    recent_searches = db.get_search_history(limit=5)
    
    if recent_searches:
        for search in recent_searches:
            if search['type'] == 'comparison' and ',' in search['query']:
                if st.sidebar.button(f"🔄 {search['query']}", key=f"recent_{search['id']}"):
                    tickers_input = search['query']
                    tickers = [ticker.strip() for ticker in tickers_input.split(",") if ticker.strip()]
                    st.session_state['compare_symbols'] = tickers_input
                    submit_button = True
    
    compare_stocks = True
    ticker = ""  # Not used in comparison mode

else:  # Watchlist mode
    # Display watchlist manager
    st.sidebar.subheader("Watchlist Options")
    
    # Time period selection for watchlist view
    selected_period = st.sidebar.selectbox("Select Time Period", list(time_periods.keys()), index=4)
    period = time_periods[selected_period]
    
    # Normalize prices option for watchlist comparison
    normalize_prices = st.sidebar.checkbox("Normalize Prices", value=True,
                                         help="Normalize all stock prices to start at 100 for easier comparison")
    
    submit_button = False  # No direct submit in watchlist mode
    compare_stocks = False
    ticker = ""
    tickers = []

# Info section about the app
st.sidebar.markdown("---")
st.sidebar.markdown("""
## About
This app fetches real-time and historical stock data from Yahoo Finance.
- Input one or more stock symbols
- Compare multiple stocks over time
- Save favorite stocks to watchlists
- View interactive charts and key metrics
- Download data as CSV
""")

# Main logic
if submit_button:
    if compare_stocks:
        # Stock comparison mode
        if len(tickers) < 2:
            st.error("Please enter at least two stock symbols for comparison.")
        else:
            # Get data for multiple stocks
            all_stock_data, all_stock_info = get_multiple_stocks_data(tickers, period)
            
            if all_stock_data and all_stock_info:
                # Display stock comparison
                st.header(f"Stock Comparison: {', '.join(tickers)}")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Price Comparison", "Performance Metrics", "Stock Data Tables"])
                
                with tab1:
                    # Create and display comparison chart
                    st.subheader(f"Price Comparison ({selected_period})")
                    
                    # Create the comparison chart
                    fig = create_comparison_chart(all_stock_data, selected_period, normalize=normalize_prices)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Allow user to toggle normalization
                    if not normalize_prices:
                        st.info("📌 For stocks with large price differences, enable 'Normalize Prices' in the sidebar for a better comparison.")
                    
                    # Calculate daily returns for correlation
                    returns_data = {}
                    for ticker, data in all_stock_data.items():
                        returns_data[ticker] = data['Close'].pct_change().dropna()
                    
                    # Create a DataFrame with all returns
                    returns_df = pd.DataFrame(returns_data)
                    
                    # Correlation heatmap
                    if len(returns_df) > 5:  # Only if we have enough data points
                        st.subheader("Price Correlation Matrix")
                        corr_matrix = returns_df.corr()
                        
                        # Create heatmap
                        fig_corr = px.imshow(
                            corr_matrix, 
                            text_auto=True, 
                            color_continuous_scale='RdBu_r',
                            title="Correlation of Daily Returns"
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Explain correlation
                        st.info("📊 The correlation matrix shows how closely the stocks move together. Values close to 1 indicate stocks that tend to move in the same direction, while values close to -1 indicate opposite movements.")
                
                with tab2:
                    st.subheader("Performance Comparison")
                    
                    # Create a performance metrics table
                    metrics_data = []
                    
                    for ticker, info in all_stock_info.items():
                        data = all_stock_data[ticker]
                        metrics = get_financial_metrics(info)
                        company_name = metrics['Company Name']
                        
                        # Calculate returns for different periods
                        price_data = data['Close']
                        current_price = price_data.iloc[-1]
                        
                        # Calculate various metrics
                        try:
                            daily_return = ((price_data.iloc[-1] / price_data.iloc[-2]) - 1) * 100 if len(price_data) > 1 else 0
                            weekly_return = ((price_data.iloc[-1] / price_data.iloc[-5]) - 1) * 100 if len(price_data) > 5 else None
                            monthly_return = ((price_data.iloc[-1] / price_data.iloc[-21]) - 1) * 100 if len(price_data) > 21 else None
                            yearly_return = ((price_data.iloc[-1] / price_data.iloc[0]) - 1) * 100
                            
                            # Calculate volatility (standard deviation of returns)
                            volatility = data['Close'].pct_change().std() * 100 * (252 ** 0.5)  # Annualized
                            
                            # Check if metrics are available
                            pe_ratio = metrics['P/E Ratio'] if metrics['P/E Ratio'] != 'N/A' else None
                            market_cap = metrics['Market Cap'] if metrics['Market Cap'] != 'N/A' else None
                            if isinstance(market_cap, (int, float)):
                                market_cap = f"${market_cap / 1_000_000_000:.2f}B"
                                
                            beta = metrics['Beta'] if metrics['Beta'] != 'N/A' else None
                            
                            metrics_data.append({
                                'Symbol': ticker,
                                'Company': company_name,
                                'Current Price': f"${current_price:.2f}",
                                'Daily Change': f"{daily_return:.2f}%" if daily_return is not None else "N/A",
                                'Weekly Change': f"{weekly_return:.2f}%" if weekly_return is not None else "N/A",
                                'Monthly Change': f"{monthly_return:.2f}%" if monthly_return is not None else "N/A",
                                'Period Return': f"{yearly_return:.2f}%",
                                'Volatility': f"{volatility:.2f}%" if not pd.isna(volatility) else "N/A",
                                'Beta': f"{beta:.2f}" if isinstance(beta, (int, float)) else "N/A",
                                'P/E Ratio': f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A",
                                'Market Cap': market_cap
                            })
                        except Exception as e:
                            st.warning(f"Could not calculate all metrics for {ticker}: {e}")
                            continue
                    
                    # Convert to DataFrame and display
                    if metrics_data:
                        performance_df = pd.DataFrame(metrics_data)
                        st.dataframe(performance_df, height=400)
                        
                        # Download performance metrics as CSV
                        csv = performance_df.to_csv(index=False)
                        st.download_button(
                            label="Download Performance Metrics as CSV",
                            data=csv,
                            file_name=f"stock_comparison_{datetime.now().strftime('%Y-%m-%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Could not generate performance metrics for comparison.")
                
                with tab3:
                    st.subheader("Stock Data Tables")
                    
                    # Let user select which stock's data to view
                    selected_ticker = st.selectbox("Select Stock to View", tickers)
                    
                    if selected_ticker in all_stock_data:
                        # Get data for the selected ticker
                        ticker_data = all_stock_data[selected_ticker]
                        ticker_info = all_stock_info[selected_ticker]
                        
                        # Format and display data
                        table_data = format_table_data(ticker_data)
                        st.dataframe(table_data, height=400)
                        
                        # Download button for CSV
                        csv = table_data.to_csv(index=False)
                        st.download_button(
                            label="Download Data as CSV",
                            data=csv,
                            file_name=f"{selected_ticker}_stock_data_{datetime.now().strftime('%Y-%m-%d')}.csv",
                            mime="text/csv"
                        )
                        
                        # Display basic statistics
                        st.subheader(f"Statistical Summary for {selected_ticker}")
                        stats = ticker_data['Close'].describe()
                        stats_df = pd.DataFrame({
                            'Statistic': stats.index,
                            'Value': stats.values
                        })
                        stats_df['Value'] = stats_df['Value'].round(2)
                        st.table(stats_df)
            else:
                st.error("Could not retrieve data for the specified stocks. Please verify the symbols and try again.")
    else:
        # Single stock analysis mode
        # Show loading spinner
        with st.spinner(f'Fetching data for {ticker}...'):
            # Get stock data
            hist_data, stock_info = get_stock_data(ticker, period)
            
            # If data was successfully retrieved
            if hist_data is not None and stock_info is not None:
                # Extract financial metrics
                metrics = get_financial_metrics(stock_info)
                company_name = metrics['Company Name']
                
                # Display company info
                st.header(f"{company_name} ({ticker})")
                
                # Create two columns for company info and current price
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Company Information")
                    st.write(f"**Sector:** {metrics['Sector']}")
                    st.write(f"**Industry:** {metrics['Industry']}")
                    st.write(f"**Market Cap:** {metrics['Market Cap']:,}" if isinstance(metrics['Market Cap'], (int, float)) else f"**Market Cap:** {metrics['Market Cap']}")
                
                with col2:
                    st.subheader("Current Trading Data")
                    
                    # Calculate price change and percentage
                    if isinstance(metrics['Current Price'], (int, float)) and isinstance(metrics['Previous Close'], (int, float)):
                        price_change = metrics['Current Price'] - metrics['Previous Close']
                        price_change_pct = (price_change / metrics['Previous Close']) * 100
                        
                        # Display with color based on price movement
                        if price_change > 0:
                            st.markdown(f"**Current Price:** ${metrics['Current Price']:.2f} <span style='color:green'>▲ ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)</span>", unsafe_allow_html=True)
                        elif price_change < 0:
                            st.markdown(f"**Current Price:** ${metrics['Current Price']:.2f} <span style='color:red'>▼ ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)</span>", unsafe_allow_html=True)
                        else:
                            st.write(f"**Current Price:** ${metrics['Current Price']:.2f} (0.00%)")
                    else:
                        st.write(f"**Current Price:** {metrics['Current Price']}")
                    
                    st.write(f"**Day Range:** ${metrics['Day Low']} - ${metrics['Day High']}" if isinstance(metrics['Day Low'], (int, float)) else f"**Day Range:** {metrics['Day Low']} - {metrics['Day High']}")
                    st.write(f"**52 Week Range:** ${metrics['52 Week Low']} - ${metrics['52 Week High']}" if isinstance(metrics['52 Week Low'], (int, float)) else f"**52 Week Range:** {metrics['52 Week Low']} - {metrics['52 Week High']}")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Price Chart", "Key Metrics", "Historical Data"])
                
                with tab1:
                    # Create and display price chart
                    fig = create_price_chart(hist_data, company_name, selected_period)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Simple moving average
                    st.subheader("Technical Indicators")
                    sma_options = [5, 10, 20, 50, 100, 200]
                    selected_smas = st.multiselect("Add Simple Moving Averages (SMA)", sma_options, default=[20, 50])
                    
                    if selected_smas:
                        # Calculate selected SMAs
                        fig_sma = go.Figure()
                        
                        # Add candlestick chart
                        fig_sma.add_trace(go.Candlestick(
                            x=hist_data.index,
                            open=hist_data['Open'],
                            high=hist_data['High'],
                            low=hist_data['Low'],
                            close=hist_data['Close'],
                            name='Price'
                        ))
                        
                        # Add SMAs
                        for period in selected_smas:
                            sma = hist_data['Close'].rolling(window=period).mean()
                            fig_sma.add_trace(go.Scatter(
                                x=hist_data.index,
                                y=sma,
                                name=f'SMA {period}',
                                line=dict(width=1.5)
                            ))
                        
                        # Update layout
                        fig_sma.update_layout(
                            title=f'{company_name} Stock Price with Moving Averages',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            height=600,
                            xaxis_rangeslider_visible=False,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig_sma, use_container_width=True)
                
                with tab2:
                    st.subheader("Key Financial Metrics")
                    
                    # Divide metrics into categories for display
                    metrics_cols = st.columns(3)
                    
                    # Trading metrics
                    with metrics_cols[0]:
                        st.markdown("##### Trading Metrics")
                        st.write(f"**Current Price:** ${metrics['Current Price']:.2f}" if isinstance(metrics['Current Price'], (int, float)) else f"**Current Price:** {metrics['Current Price']}")
                        st.write(f"**Previous Close:** ${metrics['Previous Close']:.2f}" if isinstance(metrics['Previous Close'], (int, float)) else f"**Previous Close:** {metrics['Previous Close']}")
                        st.write(f"**Open:** ${metrics['Open']:.2f}" if isinstance(metrics['Open'], (int, float)) else f"**Open:** {metrics['Open']}")
                        st.write(f"**Day Low/High:** ${metrics['Day Low']:.2f} - ${metrics['Day High']:.2f}" if isinstance(metrics['Day Low'], (int, float)) else f"**Day Low/High:** {metrics['Day Low']} - {metrics['Day High']}")
                        st.write(f"**52 Week Low/High:** ${metrics['52 Week Low']:.2f} - ${metrics['52 Week High']:.2f}" if isinstance(metrics['52 Week Low'], (int, float)) else f"**52 Week Low/High:** {metrics['52 Week Low']} - {metrics['52 Week High']}")
                        st.write(f"**Volume:** {metrics['Volume']:,}" if isinstance(metrics['Volume'], (int, float)) else f"**Volume:** {metrics['Volume']}")
                        st.write(f"**Avg Volume:** {metrics['Avg Volume']:,}" if isinstance(metrics['Avg Volume'], (int, float)) else f"**Avg Volume:** {metrics['Avg Volume']}")
                    
                    # Valuation metrics
                    with metrics_cols[1]:
                        st.markdown("##### Valuation Metrics")
                        st.write(f"**Market Cap:** ${metrics['Market Cap']:,}" if isinstance(metrics['Market Cap'], (int, float)) else f"**Market Cap:** {metrics['Market Cap']}")
                        st.write(f"**P/E Ratio:** {metrics['P/E Ratio']:.2f}" if isinstance(metrics['P/E Ratio'], (int, float)) else f"**P/E Ratio:** {metrics['P/E Ratio']}")
                        st.write(f"**EPS:** ${metrics['EPS']:.2f}" if isinstance(metrics['EPS'], (int, float)) else f"**EPS:** {metrics['EPS']}")
                        st.write(f"**Forward P/E:** {metrics['Forward P/E']:.2f}" if isinstance(metrics['Forward P/E'], (int, float)) else f"**Forward P/E:** {metrics['Forward P/E']}")
                        st.write(f"**Beta:** {metrics['Beta']:.2f}" if isinstance(metrics['Beta'], (int, float)) else f"**Beta:** {metrics['Beta']}")
                    
                    # Additional metrics
                    with metrics_cols[2]:
                        st.markdown("##### Additional Metrics")
                        st.write(f"**Dividend Yield:** {metrics['Dividend Yield']}")
                        st.write(f"**Target Price:** ${metrics['Target Price']:.2f}" if isinstance(metrics['Target Price'], (int, float)) else f"**Target Price:** {metrics['Target Price']}")
                        
                        # Get recommendations if available
                        try:
                            recommendations = yf.Ticker(ticker).recommendations
                            if recommendations is not None and not recommendations.empty:
                                recent_rec = recommendations.iloc[-1]
                                st.write(f"**Latest Recommendation:** {recent_rec['To Grade']} by {recent_rec['Firm']} ({recent_rec.name.strftime('%Y-%m-%d')})")
                        except:
                            pass
                
                with tab3:
                    st.subheader("Historical Stock Data")
                    
                    # Format data for display
                    table_data = format_table_data(hist_data)
                    
                    # Display data table
                    st.dataframe(table_data, height=400)
                    
                    # Download button for CSV
                    csv = table_data.to_csv(index=False)
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv,
                        file_name=f"{ticker}_stock_data_{datetime.now().strftime('%Y-%m-%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Display basic statistics
                    st.subheader("Statistical Summary")
                    stats = hist_data['Close'].describe()
                    stats_df = pd.DataFrame({
                        'Statistic': stats.index,
                        'Value': stats.values
                    })
                    stats_df['Value'] = stats_df['Value'].round(2)
                    st.table(stats_df)
                    
                    # Calculate returns
                    if len(hist_data) > 1:
                        hist_data['Daily Return'] = hist_data['Close'].pct_change() * 100
                        
                        # Show distribution of daily returns
                        st.subheader("Daily Returns Distribution")
                        
                        # Create histogram
                        fig_returns = go.Figure()
                        fig_returns.add_trace(go.Histogram(
                            x=hist_data['Daily Return'].dropna(),
                            nbinsx=50,
                            marker_color='blue',
                            opacity=0.7
                        ))
                        
                        # Add vertical line at mean
                        mean_return = hist_data['Daily Return'].mean()
                        fig_returns.add_vline(
                            x=mean_return,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Mean: {mean_return:.2f}%",
                            annotation_position="top right"
                        )
                        
                        fig_returns.update_layout(
                            title="Distribution of Daily Returns (%)",
                            xaxis_title="Daily Return (%)",
                            yaxis_title="Frequency",
                            height=400
                        )
                        
                        st.plotly_chart(fig_returns, use_container_width=True)
            else:
                st.error(f"Could not retrieve data for {ticker}. Please verify the stock symbol and try again.")

elif analysis_type == "Watchlists":
    # Display watchlist manager
    display_watchlist_manager()

else:
    # Display a welcome message when no symbol is entered yet
    if analysis_type == "Single Stock Analysis":
        welcome_message = "👈 Enter a stock symbol in the sidebar and click 'Get Stock Data' to begin"
    elif analysis_type == "Stock Comparison":
        welcome_message = "👈 Enter multiple stock symbols in the sidebar and click 'Compare Stocks' to begin"
    else:
        welcome_message = "👈 Use the Watchlist Manager to create and view stock watchlists"
    
    st.info(welcome_message)
    
    # Display recent searches (for the welcome screen)
    if analysis_type != "Watchlists":
        recent_searches = db.get_search_history(limit=10)
        if recent_searches:
            st.subheader("Recent Searches")
            
            search_cols = st.columns(2)
            
            single_searches = [s for s in recent_searches if s['type'] == 'single']
            comparison_searches = [s for s in recent_searches if s['type'] == 'comparison']
            
            with search_cols[0]:
                if single_searches:
                    st.markdown("##### Single Stock Searches")
                    for search in single_searches[:5]:
                        timestamp = search['timestamp'].strftime("%Y-%m-%d %H:%M")
                        if st.button(f"📊 {search['query']}", key=f"main_recent_{search['id']}"):
                            st.session_state['ticker'] = search['query']
                            st.session_state['analysis_type'] = "Single Stock Analysis"
                            st.rerun()
            
            with search_cols[1]:
                if comparison_searches:
                    st.markdown("##### Comparison Searches")
                    for search in comparison_searches[:5]:
                        timestamp = search['timestamp'].strftime("%Y-%m-%d %H:%M")
                        if st.button(f"🔄 {search['query']}", key=f"main_comp_{search['id']}"):
                            st.session_state['compare_symbols'] = search['query']
                            st.session_state['analysis_type'] = "Stock Comparison"
                            st.rerun()
    
    # Display a placeholder visualization
    st.subheader("Welcome to Stock Data Visualization Tool")
    
    # Use plotly to create a placeholder chart
    placeholder_dates = pd.date_range(end=datetime.now(), periods=100).to_list()
    # Create sample data for illustration purposes only
    placeholder_values = [None] * 100  # Initialize with None values
    placeholder_fig = go.Figure()
    placeholder_fig.add_trace(go.Scatter(
        x=placeholder_dates,
        y=placeholder_values,
        mode='lines',
        name='Stocks',
        line=dict(color='lightgray', dash='dash')
    ))
    placeholder_fig.update_layout(
        title="Stock data will appear here",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        annotations=[dict(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Enter stock symbol(s) to view real data",
            showarrow=False,
            font=dict(size=20)
        )]
    )
    st.plotly_chart(placeholder_fig, use_container_width=True)
    
    # Add feature highlights
    st.markdown("### 📊 Features")
    
    feature_cols = st.columns(3)
    
    with feature_cols[0]:
        st.markdown("**Single Stock Analysis**")
        st.markdown("- Interactive price charts with candlestick patterns")
        st.markdown("- Key financial metrics and company information")
        st.markdown("- Technical indicators including SMAs")
        st.markdown("- Historical data available for download")
    
    with feature_cols[1]:
        st.markdown("**Stock Comparison**")
        st.markdown("- Compare multiple stocks on the same chart")
        st.markdown("- Normalized price view for easier comparison")
        st.markdown("- Correlation analysis between stocks")
        st.markdown("- Performance metrics table for all stocks")
        
    with feature_cols[2]:
        st.markdown("**Watchlists**")
        st.markdown("- Create custom watchlists")
        st.markdown("- Save favorite stocks for quick access")
        st.markdown("- Compare all stocks in a watchlist")
        st.markdown("- Track performance across your portfolio")
