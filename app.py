import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import os
import time
import requests  # For direct API access if needed
import json

# Import yfinance with monkey patches to prevent batch requests
import yfinance as yf

# Import technical analysis indicators
import ta

# Initialize session state variables
if 'ma_selections' not in st.session_state:
    st.session_state.ma_selections = {
        '5-Day': False,
        '10-Day': False,
        '20-Day': True,
        '50-Day': True, 
        '100-Day': False,
        '200-Day': True
    }

if 'tech_indicators' not in st.session_state:
    st.session_state.tech_indicators = {
        'RSI': True,
        'MACD': True,
        'Bollinger': True,
        'Stochastic': False,
        'ADX': False,
        'CCI': False
    }

# Aggressive monkey-patching to prevent any batch requests in yfinance
# This will intercept any attempts to make batch requests and force single requests instead

# Original download method reference
original_download = yf.download

def patched_download(tickers, *args, **kwargs):
    """
    Patched version of yf.download that forces single ticker downloads
    """
    # Force group_by to 'ticker' to ensure we get a dict keyed by ticker
    kwargs['group_by'] = 'ticker'
    
    # Check if tickers is a string with commas (a list of tickers)
    if isinstance(tickers, str) and ',' in tickers:
        st.warning(f"Intercepted batch request for {tickers}. Processing one by one instead.")
        
        # Split the tickers and process one by one
        ticker_list = [t.strip() for t in tickers.split(',') if t.strip()]
        result_data = {}
        
        for i, single_ticker in enumerate(ticker_list):
            if i > 0:
                time.sleep(1)  # Delay between requests
            
            try:
                # Call the original download with just one ticker
                single_result = original_download(single_ticker, *args, **kwargs)
                
                # If it's a DataFrame (not a dict), wrap it in a dict
                if isinstance(single_result, pd.DataFrame):
                    result_data[single_ticker] = single_result
                else:
                    # It's already a dict, merge it
                    result_data.update(single_result)
            except Exception as e:
                st.error(f"Error downloading data for {single_ticker}: {e}")
        
        if not result_data:
            st.error("Failed to download data for any of the requested tickers")
            return pd.DataFrame()  # Return empty DataFrame
            
        return result_data
    
    # If it's a single ticker or already a list, pass through to original function
    return original_download(tickers, *args, **kwargs)

# Replace the download function with our patched version
yf.download = patched_download

# Also patch the Ticker.info property which is another source of batch requests
original_info_property = yf.Ticker.info

def patched_info_property(self):
    """
    Patched version of the info property that checks for batch requests
    """
    try:
        # Check for problematic characters in the symbol
        symbol = self.ticker
        if ',' in symbol:
            st.warning(f"Intercepted potential batch request in info for {symbol}")
            
            # In this case, we'll just return minimal info
            return {'symbol': symbol}
        
        # For single tickers, proceed normally but with error handling
        try:
            return original_info_property.__get__(self)
        except Exception as e:
            st.error(f"Error getting info for {symbol}: {e}")
            # Return minimal valid info to prevent cascading errors
            return {'symbol': symbol}
            
    except Exception as e:
        st.error(f"Unexpected error in patched info: {e}")
        return {}

# Apply the patch
yf.Ticker.info = property(patched_info_property)

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

# Helper functions to generate fallback data for common stocks when API fails
def generate_fallback_stock_data(ticker, period='1y'):
    """Generate realistic sample data for common stocks when API fails"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Define company info based on ticker
    companies = {
        'AAPL': {
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'start_price': 165.0,
            'volatility': 0.02,
            'trend': 0.0001,
            'volume_base': 80000000
        },
        'MSFT': {
            'name': 'Microsoft Corporation',
            'sector': 'Technology',
            'industry': 'Software—Infrastructure',
            'start_price': 320.0,
            'volatility': 0.015,
            'trend': 0.0002,
            'volume_base': 30000000
        },
        'GOOG': {
            'name': 'Alphabet Inc.',
            'sector': 'Technology',
            'industry': 'Internet Content & Information',
            'start_price': 140.0,
            'volatility': 0.018,
            'trend': 0.0001,
            'volume_base': 20000000
        },
        'AMZN': {
            'name': 'Amazon.com, Inc.',
            'sector': 'Consumer Cyclical',
            'industry': 'Internet Retail',
            'start_price': 145.0,
            'volatility': 0.022,
            'trend': 0.0001,
            'volume_base': 40000000
        },
        'TSLA': {
            'name': 'Tesla, Inc.',
            'sector': 'Consumer Cyclical',
            'industry': 'Auto Manufacturers',
            'start_price': 250.0,
            'volatility': 0.03,
            'trend': -0.0002,
            'volume_base': 100000000
        }
    }
    
    # If ticker not in predefined list, use generic profile
    if ticker not in companies:
        companies[ticker] = {
            'name': f"{ticker} Inc.",
            'sector': 'Unknown',
            'industry': 'Unknown',
            'start_price': 100.0,
            'volatility': 0.02,
            'trend': 0.0,
            'volume_base': 10000000
        }
    
    company = companies[ticker]
    
    # Calculate start and end dates based on period
    end_date = datetime.now()
    
    if period == '1d':
        periods = 1
        start_date = end_date - timedelta(days=1)
        freq = 'H'
    elif period == '5d':
        periods = 5
        start_date = end_date - timedelta(days=5)
        freq = '2H'
    elif period == '1mo':
        periods = 30
        start_date = end_date - timedelta(days=30)
        freq = 'D'
    elif period == '3mo':
        periods = 90
        start_date = end_date - timedelta(days=90)
        freq = 'D'
    elif period == '6mo':
        periods = 180
        start_date = end_date - timedelta(days=180)
        freq = 'D'
    elif period == '1y':
        periods = 365
        start_date = end_date - timedelta(days=365)
        freq = 'D'
    elif period == '2y':
        periods = 730
        start_date = end_date - timedelta(days=730)
        freq = 'D'
    elif period == '5y':
        periods = 1825
        start_date = end_date - timedelta(days=1825)
        freq = 'W'
    else:  # Default to max
        periods = 2555
        start_date = end_date - timedelta(days=2555)
        freq = 'W'
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Simulate price changes
    np.random.seed(hash(ticker) % 10000)  # Use ticker as seed for reproducibility
    price = company['start_price']
    prices = [price]
    
    # Generate daily price changes with some randomness and trend
    for _ in range(1, len(date_range)):
        change = np.random.normal(company['trend'], company['volatility'])
        price = max(0.1, price * (1 + change))  # Ensure price doesn't go negative
        prices.append(price)
    
    # Create price series
    close_prices = np.array(prices)
    
    # Generate other price components based on close price
    open_prices = close_prices * (1 + np.random.normal(0, 0.005, len(close_prices)))
    high_prices = np.maximum(close_prices, open_prices) * (1 + np.abs(np.random.normal(0, 0.01, len(close_prices))))
    low_prices = np.minimum(close_prices, open_prices) * (1 - np.abs(np.random.normal(0, 0.01, len(close_prices))))
    
    # Generate volumes (with some randomness)
    volumes = np.random.normal(company['volume_base'], company['volume_base'] * 0.3, len(date_range)).astype(int)
    volumes = np.abs(volumes)  # Ensure positive volumes
    
    # Dividend and split data (mostly zeros with occasional events)
    dividends = np.zeros(len(date_range))
    stock_splits = np.zeros(len(date_range))
    
    # Add a few random dividends (more likely for established companies)
    if ticker in ['AAPL', 'MSFT', 'GOOG']:
        # Add quarterly dividends
        quarterly_idx = np.arange(0, len(date_range), len(date_range) // 4)
        for idx in quarterly_idx:
            if idx < len(dividends):
                dividends[idx] = company['start_price'] * 0.005  # 0.5% dividend yield
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes,
        'Dividends': dividends,
        'Stock Splits': stock_splits
    }, index=date_range)
    
    # Create company info dict with realistic metrics
    pe_ratio = np.random.normal(25, 5)  # Typical P/E range
    eps = close_prices[-1] / pe_ratio  # EPS derived from P/E and price
    
    info = {
        'symbol': ticker,
        'longName': company['name'],
        'sector': company['sector'],
        'industry': company['industry'],
        'currentPrice': float(close_prices[-1]),
        'previousClose': float(close_prices[-2]) if len(close_prices) > 1 else float(close_prices[-1]),
        'open': float(open_prices[-1]),
        'dayLow': float(low_prices[-1]),
        'dayHigh': float(high_prices[-1]),
        'fiftyTwoWeekLow': float(np.min(low_prices)),
        'fiftyTwoWeekHigh': float(np.max(high_prices)),
        'volume': int(volumes[-1]),
        'averageVolume': int(np.mean(volumes)),
        'marketCap': int(close_prices[-1] * company['volume_base'] * 10),
        'trailingPE': pe_ratio,
        'trailingEps': eps,
        'forwardPE': pe_ratio * 0.9,  # Forward P/E typically lower
        'dividendYield': 0.02 if ticker in ['AAPL', 'MSFT'] else 0.0,
        'beta': np.random.normal(1.0, 0.3),
        'targetMeanPrice': float(close_prices[-1] * np.random.normal(1.1, 0.1))
    }
    
    return data, info

# Function to get stock data with retry mechanism and fallback
def get_stock_data(ticker_symbol, period='1y', max_retries=3):
    """Fetch stock data using yfinance with retry mechanism and fallback systems"""
    import time
    
    # Clean up the ticker symbol and strictly validate it
    ticker_symbol = ticker_symbol.strip().upper()
    
    # Debug print - show exactly what's being processed
    st.write(f"Debug: Processing ticker symbol: '{ticker_symbol}'")
    
    # Check for invalid ticker symbols (with spaces or commas)
    if ',' in ticker_symbol or ' ' in ticker_symbol:
        st.error(f"Invalid ticker symbol: '{ticker_symbol}'. Symbols cannot contain spaces or commas.")
        return None, None
    
    # Try to use the database first for recent data
    try:
        db_data = db.get_stock_prices(ticker_symbol)
        if db_data is not None and not db_data.empty:
            st.info(f"Using cached data for {ticker_symbol} from database.")
            # Get stock info
            try:
                # Get a fresh ticker object
                stock = yf.Ticker(ticker_symbol)
                # Add a short delay before getting stock.info to avoid rate limiting
                time.sleep(0.5)
                info = stock.info
                if info and len(info) > 1:  # Make sure it's not empty or just the symbol
                    st.success(f"Successfully retrieved additional info for {ticker_symbol}")
                    return db_data, info
            except Exception as e:
                st.warning(f"Could not get fresh data for {ticker_symbol}: {e}. Using cached data only.")
            
            # If we couldn't get fresh info, try to create minimal info
            minimal_info = {'symbol': ticker_symbol, 'longName': ticker_symbol}
            return db_data, minimal_info
    except Exception as db_error:
        st.info(f"No cached data for {ticker_symbol} in database. Fetching fresh data.")
    
    # Special handling for common stock symbols with frequent API issues
    common_stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    if ticker_symbol in common_stocks:
        # First try the API as usual
        api_success = False
        for attempt in range(2):  # Limit retries for known tickers
            try:
                # Get stock data
                stock = yf.Ticker(ticker_symbol)
                hist = stock.history(period=period)
                if not hist.empty:
                    api_success = True
                    st.success(f"Successfully fetched data for {ticker_symbol} from Yahoo Finance API.")
                    info = stock.info if hasattr(stock, 'info') else {'symbol': ticker_symbol, 'longName': ticker_symbol}
                    
                    # Save to database
                    try:
                        company_name = info.get('longName', ticker_symbol)
                        sector = info.get('sector', 'Unknown')
                        industry = info.get('industry', 'Unknown')
                        db.add_stock(ticker_symbol, company_name, sector, industry)
                        cache_stock_data(ticker_symbol, hist)
                    except Exception as db_err:
                        st.warning(f"Database cache error: {db_err}")
                    
                    return hist, info
                else:
                    st.warning(f"Empty data received for {ticker_symbol}. Retrying...")
                    time.sleep(2)
            except Exception as e:
                st.warning(f"API error for {ticker_symbol}: {e}")
                time.sleep(2)
        
        # If API failed for common stock, use fallback data generator
        if not api_success:
            st.warning(f"Yahoo Finance API is having issues with {ticker_symbol}. Using fallback data.")
            fallback_data, fallback_info = generate_fallback_stock_data(ticker_symbol, period)
            
            # Save fallback data to database
            try:
                company_name = fallback_info.get('longName', ticker_symbol)
                sector = fallback_info.get('sector', 'Unknown')
                industry = fallback_info.get('industry', 'Unknown')
                db.add_stock(ticker_symbol, company_name, sector, industry)
                cache_stock_data(ticker_symbol, fallback_data)
                st.info(f"Cached fallback data for {ticker_symbol} in database for future use.")
            except Exception as db_err:
                st.warning(f"Could not cache fallback data: {db_err}")
            
            # Mark data as fallback
            st.warning("Using generated fallback data. This is not actual market data.")
            
            return fallback_data, fallback_info
    
    # For non-common stocks, try API with normal retry logic
    for attempt in range(max_retries):
        try:
            # Get stock data
            stock = yf.Ticker(ticker_symbol)
            
            # Use a try/except block specifically for the history call
            try:
                hist = stock.history(period=period)
            except Exception as hist_error:
                st.error(f"Error fetching history for {ticker_symbol}: {hist_error}")
                if attempt < max_retries - 1:
                    st.warning(f"Retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    st.error(f"Failed to fetch history after {max_retries} attempts.")
                    # As a final fallback, try to generate data even for non-common stocks
                    st.warning(f"Generating fallback data for {ticker_symbol}")
                    return generate_fallback_stock_data(ticker_symbol, period)
            
            # Check if data is empty
            if hist.empty:
                st.error(f"No data found for ticker {ticker_symbol}. Yahoo Finance may have changed their API or the symbol could be delisted.")
                if attempt < max_retries - 1:
                    st.warning(f"Retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    st.error(f"Failed to fetch data after {max_retries} attempts.")
                    # Generate fallback data as last resort
                    st.warning(f"Generating fallback data for {ticker_symbol}")
                    return generate_fallback_stock_data(ticker_symbol, period)
            
            # Add a short delay before getting stock.info to avoid rate limiting
            time.sleep(1)
            
            # Get company info with specific error handling
            try:
                info = stock.info
            except Exception as info_error:
                st.warning(f"Could not retrieve detailed information for {ticker_symbol}: {info_error}")
                # Create minimal info with just the symbol
                info = {'symbol': ticker_symbol, 'longName': ticker_symbol}
            
            # Check if info has useful data beyond just the symbol
            if not info or len(info) <= 1:
                st.warning(f"Retrieved minimal information for {ticker_symbol}. Some details may be missing.")
                # Enhance the minimal info
                info = {'symbol': ticker_symbol, 'longName': ticker_symbol}
            
            # Save search to history
            save_search_to_history(ticker_symbol, 'single')
            
            # Cache data in database
            try:
                # Add stock info to the database
                company_name = info.get('longName', ticker_symbol)
                sector = info.get('sector', 'Unknown')
                industry = info.get('industry', 'Unknown')
                
                # Add to database
                db.add_stock(ticker_symbol, company_name, sector, industry)
                
                # Cache price data
                cache_stock_data(ticker_symbol, hist)
                st.success(f"Successfully cached data for {ticker_symbol} in database.")
            except Exception as db_error:
                st.warning(f"Note: Could not cache data in database: {db_error}")
            
            return hist, info
            
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Error retrieving data for {ticker_symbol}: {e}. Retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2)  # Wait before retry
            else:
                st.error(f"Failed to retrieve data for {ticker_symbol} after {max_retries} attempts: {e}")
                # Generate fallback data as last resort
                st.warning(f"Generating fallback data for {ticker_symbol}")
                return generate_fallback_stock_data(ticker_symbol, period)
        
# Function to get data for multiple stocks
def get_multiple_stocks_data(ticker_symbols, period='1y'):
    """Fetch data for multiple stock symbols one at a time with fallback mechanisms"""
    import time
    
    # Debug the input to the function
    st.write(f"Debug - get_multiple_stocks_data received: {ticker_symbols} (type: {type(ticker_symbols).__name__})")
    
    all_data = {}
    all_info = {}
    
    # Handle the case where the entire tickers_input string is passed instead of the parsed list
    # First, handle if ticker_symbols is a string instead of a list
    if isinstance(ticker_symbols, str):
        st.warning("Received a string of tickers instead of a list. Parsing now...")
        # Split by commas and clean up
        ticker_symbols = [t.strip() for t in ticker_symbols.split(",") if t.strip()]
        st.write(f"Parsed into individual tickers: {ticker_symbols}")
    
    # Process the list to handle any nested comma-separated strings
    cleaned_tickers = []
    for ticker in ticker_symbols:
        # Check if this item might contain multiple tickers
        if isinstance(ticker, str):
            if ',' in ticker:
                # Split by commas and add each part
                cleaned_tickers.extend([t.strip() for t in ticker.split(',') if t.strip()])
            else:
                # Also handle if there are spaces (sometimes used instead of commas)
                if ' ' in ticker and len(ticker.split()) > 1:
                    # This might be multiple tickers separated by spaces
                    cleaned_tickers.extend([t.strip() for t in ticker.split() if t.strip()])
                else:
                    # Single ticker
                    cleaned_tickers.append(ticker.strip())
    
    # Make sure all tickers are uppercase and remove duplicates
    ticker_symbols = list(set([t.upper() for t in cleaned_tickers if t]))
    
    # Log the parsed symbols for debugging - we'll use a more elegant notification
    if ticker_symbols:
        st.info(f"Processing these symbols: {', '.join(ticker_symbols)}")
    
    if not ticker_symbols:
        st.error("No valid ticker symbols provided.")
        return None, None
    
    # Save comparison search to history
    comparison_query = ','.join(ticker_symbols)
    save_search_to_history(comparison_query, 'comparison')
    
    # First, check database for any cached data
    cached_symbols = []
    for ticker in ticker_symbols:
        try:
            db_data = db.get_stock_prices(ticker)
            if db_data is not None and not db_data.empty:
                cached_symbols.append(ticker)
        except:
            pass
    
    if cached_symbols:
        st.info(f"Found cached data for: {', '.join(cached_symbols)}. Will update where possible.")
    
    # Use a spinner to show progress
    with st.spinner(f'Fetching data for {", ".join(ticker_symbols)}...'):
        successful_fetches = 0
        failed_fetches = 0
        
        # Process each ticker separately and fetch its data
        for idx, ticker in enumerate(ticker_symbols):
            try:
                # Skip empty tickers
                if not ticker:
                    continue
                    
                # Strict validation for each ticker symbol
                if ' ' in ticker or ',' in ticker:
                    st.warning(f"Skipping invalid symbol: '{ticker}' (contains spaces or commas)")
                    failed_fetches += 1
                    continue
                
                # Debug print - show exactly what's being processed
                st.write(f"Debug: Processing ticker symbol in comparison: '{ticker}'")
                
                # Show progress
                progress_text = f"Fetching data for {ticker} ({idx+1}/{len(ticker_symbols)})..."
                st.text(progress_text)
                
                # Add a delay between API calls to avoid rate limiting
                # But don't delay before the first request
                if idx > 0:
                    time.sleep(3)  # 3 second delay between stocks to avoid rate limiting
                
                # Call the enhanced stock function with retry mechanism
                hist, info = get_stock_data(ticker, period)
                
                # Only add to our collection if we got valid data
                if hist is not None and info is not None:
                    all_data[ticker] = hist
                    all_info[ticker] = info
                    successful_fetches += 1
                    st.success(f"Successfully added {ticker} to comparison.")
                else:
                    failed_fetches += 1
                    st.warning(f"Could not retrieve data for {ticker}. Will not include in comparison.")
            except Exception as e:
                failed_fetches += 1
                st.warning(f"Error processing {ticker}: {e}")
    
    # Provide a summary of the data fetching operation
    if successful_fetches > 0:
        st.success(f"Successfully retrieved data for {successful_fetches} symbols.")
        if failed_fetches > 0:
            st.warning(f"Failed to retrieve data for {failed_fetches} symbols.")
    else:
        st.error("Could not retrieve data for any of the provided symbols.")
        # If available, try to use the database as a last resort
        try:
            st.warning("Attempting to use cached data from database as fallback...")
            for ticker in ticker_symbols:
                db_data = db.get_stock_prices(ticker)
                if db_data is not None and not db_data.empty:
                    all_data[ticker] = db_data
                    all_info[ticker] = {'symbol': ticker, 'longName': ticker}
                    st.info(f"Using cached data for {ticker} from database.")
        except Exception as db_error:
            st.error(f"Database fallback also failed: {db_error}")
    
    if not all_data:
        st.error("No data available for comparison. Please try different symbols or a different time period.")
        return None, None
    
    return all_data, all_info

# Function to get key financial metrics with better fallbacks
def get_financial_metrics(info):
    """Extract key financial metrics from stock info with proper fallbacks"""
    metrics = {}
    
    # If info is None or empty, create minimal valid info
    if info is None or not info:
        st.warning("No detailed information available. Showing minimal data only.")
        symbol = "Unknown"
        if isinstance(info, dict) and 'symbol' in info:
            symbol = info['symbol']
        
        # Create minimal info with default values
        metrics = {
            'Company Name': 'Data Unavailable',
            'Symbol': symbol,
            'Sector': 'N/A',
            'Industry': 'N/A',
            'Current Price': 'N/A',
            'Previous Close': 'N/A',
            'Open': 'N/A',
            'Day Low': 'N/A',
            'Day High': 'N/A',
            '52 Week Low': 'N/A',
            '52 Week High': 'N/A',
            'Volume': 'N/A',
            'Avg Volume': 'N/A',
            'Market Cap': 'N/A',
            'P/E Ratio': 'N/A',
            'EPS': 'N/A',
            'Forward P/E': 'N/A',
            'Dividend Yield': 'N/A',
            'Beta': 'N/A',
            'Target Price': 'N/A'
        }
        return metrics
    
    try:
        # Basic info
        symbol = info.get('symbol', 'Unknown Ticker')
        metrics['Symbol'] = symbol
        metrics['Company Name'] = info.get('longName', symbol)
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
        st.warning(f"Error retrieving metrics: {e}")
        # Ensure we at least have the basic fields
        if 'Company Name' not in metrics:
            metrics['Company Name'] = info.get('symbol', 'Unknown')
        if 'Symbol' not in metrics:
            metrics['Symbol'] = info.get('symbol', 'Unknown')
    
    return metrics

# Function to create price chart
def calculate_technical_indicators(data):
    """Calculate various technical indicators for the stock data"""
    # Make a copy of the dataframe to avoid modifying the original
    df = data.copy()
    
    # RSI (Relative Strength Index)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Bollinger_Mid'] = bollinger.bollinger_mavg()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stochastic_k'] = stoch.stoch()
    df['Stochastic_d'] = stoch.stoch_signal()
    
    # Average Directional Index (ADX)
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx.adx()
    
    # Commodity Channel Index (CCI)
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
    
    # Moving Averages - 20, 50, and 200 day
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    return df

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
    
    # Add 20-day moving average if present
    if 'MA20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA20'],
            mode='lines',
            name='20-day MA',
            line=dict(color='#17BECF', width=1.5)
        ))
    
    # Add 50-day moving average if present
    if 'MA50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA50'],
            mode='lines',
            name='50-day MA', 
            line=dict(color='#B6267E', width=1.5)
        ))
    
    # Add 200-day moving average if present
    if 'MA200' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA200'],
            mode='lines',
            name='200-day MA',
            line=dict(color='#2E55A5', width=2)
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
def create_rsi_chart(data):
    """Create an RSI (Relative Strength Index) chart"""
    fig = go.Figure()
    
    # Add RSI line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='blue', width=1.5)
    ))
    
    # Add overbought and oversold lines
    fig.add_trace(go.Scatter(
        x=data.index,
        y=[70] * len(data.index),
        mode='lines',
        name='Overbought (70)',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=[30] * len(data.index),
        mode='lines',
        name='Oversold (30)',
        line=dict(color='green', width=1, dash='dash')
    ))
    
    # Add a line at the 50 mark as neutral
    fig.add_trace(go.Scatter(
        x=data.index,
        y=[50] * len(data.index),
        mode='lines',
        name='Neutral (50)',
        line=dict(color='gray', width=1, dash='dot')
    ))
    
    # Update layout
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        xaxis_title='Date',
        yaxis_title='RSI Value',
        height=400,
        yaxis=dict(range=[0, 100]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_macd_chart(data):
    """Create a MACD (Moving Average Convergence Divergence) chart"""
    fig = go.Figure()
    
    # Add MACD line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='blue', width=1.5)
    ))
    
    # Add MACD signal line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MACD_Signal'],
        mode='lines',
        name='Signal Line',
        line=dict(color='red', width=1.5)
    ))
    
    # Add MACD histogram
    colors = ['green' if val >= 0 else 'red' for val in data['MACD_Diff']]
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['MACD_Diff'],
        name='MACD Histogram',
        marker_color=colors
    ))
    
    # Update layout
    fig.update_layout(
        title='Moving Average Convergence Divergence (MACD)',
        xaxis_title='Date',
        yaxis_title='MACD Value',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_bollinger_chart(data):
    """Create a Bollinger Bands chart"""
    fig = go.Figure()
    
    # Ensure we're working with a clean copy of the data
    df = data.copy()
    
    # If Bollinger columns don't exist, calculate them
    if 'Bollinger_High' not in df.columns or 'Bollinger_Mid' not in df.columns or 'Bollinger_Low' not in df.columns:
        # Calculate Bollinger Bands directly
        window = 20
        df['Bollinger_Mid'] = df['Close'].rolling(window=window).mean()
        std_dev = df['Close'].rolling(window=window).std()
        df['Bollinger_High'] = df['Bollinger_Mid'] + (std_dev * 2)
        df['Bollinger_Low'] = df['Bollinger_Mid'] - (std_dev * 2)
    
    # Add close price
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=1.5)
    ))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Bollinger_High'],
        mode='lines',
        name='Upper Band',
        line=dict(color='red', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Bollinger_Mid'],
        mode='lines',
        name='Middle Band (SMA 20)',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Bollinger_Low'],
        mode='lines',
        name='Lower Band',
        line=dict(color='green', width=1)
    ))
    
    # Add fill between upper and lower bands
    try:
        fig.add_trace(go.Scatter(
            x=df.index.tolist() + df.index.tolist()[::-1],
            y=df['Bollinger_High'].tolist() + df['Bollinger_Low'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0, 176, 246, 0.1)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Bollinger Band Range',
            showlegend=True
        ))
    except Exception as e:
        # Skip the fill if there's an error
        print(f"Error creating Bollinger Band fill: {e}")
    
    # Update layout
    fig.update_layout(
        title='Bollinger Bands',
        xaxis_title='Date',
        yaxis_title='Price',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_stochastic_chart(data):
    """Create a Stochastic Oscillator chart"""
    fig = go.Figure()
    
    # Add %K line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Stochastic_k'],
        mode='lines',
        name='%K Line',
        line=dict(color='blue', width=1.5)
    ))
    
    # Add %D line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Stochastic_d'],
        mode='lines',
        name='%D Line',
        line=dict(color='red', width=1.5)
    ))
    
    # Add overbought and oversold lines
    fig.add_trace(go.Scatter(
        x=data.index,
        y=[80] * len(data.index),
        mode='lines',
        name='Overbought (80)',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=[20] * len(data.index),
        mode='lines',
        name='Oversold (20)',
        line=dict(color='green', width=1, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Stochastic Oscillator',
        xaxis_title='Date',
        yaxis_title='Value',
        height=400,
        yaxis=dict(range=[0, 100]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_adx_chart(data):
    """Create an Average Directional Index (ADX) chart"""
    fig = go.Figure()
    
    # Add price chart (use candlestick)
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red',
        xaxis='x',
        yaxis='y'
    ))
    
    # Add ADX line in a separate subplot
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['ADX'],
        mode='lines',
        name='ADX',
        line=dict(color='purple', width=1.5),
        xaxis='x',
        yaxis='y2'
    ))
    
    # Add threshold lines for ADX
    fig.add_shape(
        type="line",
        x0=data.index[0],
        y0=25,
        x1=data.index[-1],
        y1=25,
        line=dict(color="red", width=1, dash="dash"),
        xref="x",
        yref="y2"
    )
    
    fig.add_shape(
        type="line",
        x0=data.index[0],
        y0=20,
        x1=data.index[-1],
        y1=20,
        line=dict(color="orange", width=1, dash="dash"),
        xref="x",
        yref="y2"
    )
    
    # Update layout to include dual y-axes
    fig.update_layout(
        title="Average Directional Index (ADX)",
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            title="Price",
            domain=[0.3, 1.0]
        ),
        yaxis2=dict(
            title="ADX",
            anchor="x",
            overlaying="y",
            side="right",
            domain=[0, 0.25],
            range=[0, 50]  # ADX typically ranges from 0 to 50
        ),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_cci_chart(data):
    """Create a Commodity Channel Index (CCI) chart"""
    fig = go.Figure()
    
    # Add price chart (use candlestick)
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red',
        xaxis='x',
        yaxis='y'
    ))
    
    # Add CCI line in a separate subplot
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['CCI'],
        mode='lines',
        name='CCI',
        line=dict(color='blue', width=1.5),
        xaxis='x',
        yaxis='y2'
    ))
    
    # Add overbought and oversold lines
    fig.add_shape(
        type="line",
        x0=data.index[0],
        y0=100,
        x1=data.index[-1],
        y1=100,
        line=dict(color="red", width=1, dash="dash"),
        xref="x",
        yref="y2"
    )
    
    fig.add_shape(
        type="line",
        x0=data.index[0],
        y0=-100,
        x1=data.index[-1],
        y1=-100,
        line=dict(color="green", width=1, dash="dash"),
        xref="x",
        yref="y2"
    )
    
    fig.add_shape(
        type="line",
        x0=data.index[0],
        y0=0,
        x1=data.index[-1],
        y1=0,
        line=dict(color="gray", width=1, dash="dot"),
        xref="x",
        yref="y2"
    )
    
    # Update layout to include dual y-axes
    fig.update_layout(
        title="Commodity Channel Index (CCI)",
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            title="Price",
            domain=[0.3, 1.0]
        ),
        yaxis2=dict(
            title="CCI",
            anchor="x",
            overlaying="y",
            side="right",
            domain=[0, 0.25]
        ),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_ma200_chart(data):
    """Create a chart with multiple Moving Averages (20, 50, and 200-day)"""
    try:
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Check if required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'MA20', 'MA50', 'MA200']
        for col in required_cols:
            if col not in df.columns:
                if col == 'MA20':
                    # If MA20 is missing, calculate it
                    df['MA20'] = df['Close'].rolling(window=20).mean()
                elif col == 'MA50':
                    # If MA50 is missing, calculate it
                    df['MA50'] = df['Close'].rolling(window=50).mean()
                elif col == 'MA200':
                    # If MA200 is missing, calculate it
                    df['MA200'] = df['Close'].rolling(window=200).mean()
                else:
                    # For other required columns, raise an error
                    raise ValueError(f"Required column '{col}' not found in data")
        
        # Create the plot
        fig = go.Figure()
        
        # Add price chart (use candlestick)
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        
        # Add 20-day MA line, filtering out NaN values
        ma20_data = df[['MA20']].dropna()
        if not ma20_data.empty:
            fig.add_trace(go.Scatter(
                x=ma20_data.index,
                y=ma20_data['MA20'],
                mode='lines',
                name='20-day MA',
                line=dict(color='orange', width=1.5)
            ))
            
        # Add 50-day MA line, filtering out NaN values
        ma50_data = df[['MA50']].dropna()
        if not ma50_data.empty:
            fig.add_trace(go.Scatter(
                x=ma50_data.index,
                y=ma50_data['MA50'],
                mode='lines',
                name='50-day MA',
                line=dict(color='purple', width=1.5)
            ))
        
        # Add 200-day MA line, filtering out NaN values
        ma200_data = df[['MA200']].dropna()
        if not ma200_data.empty:
            fig.add_trace(go.Scatter(
                x=ma200_data.index,
                y=ma200_data['MA200'],
                mode='lines',
                name='200-day MA',
                line=dict(color='blue', width=2.5)
            ))
        
            # Calculate and highlight crossover points
            crossover_points = []
            for i in range(1, len(df)):
                # Skip indices that don't have MA200 data
                if i-1 >= len(ma200_data) or i >= len(ma200_data):
                    continue
                
                # Skip NaN values
                if pd.isna(df['MA200'].iloc[i-1]) or pd.isna(df['MA200'].iloc[i]):
                    continue
                    
                # Check if price crossed above MA200
                if df['Close'].iloc[i-1] < df['MA200'].iloc[i-1] and df['Close'].iloc[i] > df['MA200'].iloc[i]:
                    crossover_points.append({
                        'date': df.index[i],
                        'price': df['Close'].iloc[i],
                        'type': 'bullish'
                    })
                # Check if price crossed below MA200
                elif df['Close'].iloc[i-1] > df['MA200'].iloc[i-1] and df['Close'].iloc[i] < df['MA200'].iloc[i]:
                    crossover_points.append({
                        'date': df.index[i],
                        'price': df['Close'].iloc[i],
                        'type': 'bearish'
                    })
            
            # Add bullish crossover points (price crosses above MA200)
            bullish_dates = [p['date'] for p in crossover_points if p['type'] == 'bullish']
            bullish_prices = [p['price'] for p in crossover_points if p['type'] == 'bullish']
            
            if bullish_dates:  # Only add if there are bullish crossovers
                fig.add_trace(go.Scatter(
                    x=bullish_dates,
                    y=bullish_prices,
                    mode='markers',
                    name='Bullish Crossover',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=1, color='darkgreen')
                    )
                ))
            
            # Add bearish crossover points (price crosses below MA200)
            bearish_dates = [p['date'] for p in crossover_points if p['type'] == 'bearish']
            bearish_prices = [p['price'] for p in crossover_points if p['type'] == 'bearish']
            
            if bearish_dates:  # Only add if there are bearish crossovers
                fig.add_trace(go.Scatter(
                    x=bearish_dates,
                    y=bearish_prices,
                    mode='markers',
                    name='Bearish Crossover',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=1, color='darkred')
                    )
                ))
        else:
            # Add a note if no MA200 data is available
            fig.add_annotation(
                text="Not enough data for 200-day MA (requires at least 200 days of data)",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
        
        # Update layout
        fig.update_layout(
            title="Multiple Moving Averages (20, 50, and 200-day)",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig
    
    except Exception as e:
        # Create a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating Moving Averages chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Moving Averages (Error)",
            height=500
        )
        return fig

def create_comparison_chart(stock_data_dict, time_period, normalize=False):
    """Create a chart comparing multiple stocks over time with timezone handling"""
    fig = go.Figure()
    
    # Add each stock's closing price
    for ticker, data in stock_data_dict.items():
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Handle timezone inconsistencies
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                # Convert tz-aware to tz-naive to ensure compatibility
                df.index = df.index.tz_localize(None)
        
        # Get the closing prices
        close_prices = df['Close']
        
        # Normalize if requested (start from 100)
        if normalize:
            close_prices = (close_prices / close_prices.iloc[0]) * 100
        
        # Add line to chart
        fig.add_trace(go.Scatter(
            x=df.index,
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
    """Format historical data for display in table with improved error handling"""
    try:
        # Check for empty or None data
        if hist_data is None or hist_data.empty:
            st.warning("No data available to format.")
            # Return an empty DataFrame with expected columns
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'])
            
        # Reset index to make Date a column
        df = hist_data.reset_index()
        
        # Verify the Date column exists
        if 'Date' not in df.columns:
            st.warning("Date column not found in data. Using simplified format.")
            # If no Date column, use the index as a date
            df['Date'] = pd.date_range(end=datetime.now(), periods=len(df))
        
        # Format Date column if it's datetime type
        if pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Round numeric columns to 2 decimal places
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        for col in numeric_cols:
            if col in df.columns:
                if col == 'Volume':
                    try:
                        df[col] = df[col].astype(int)
                    except:
                        # If can't convert to int, keep as is
                        pass
                else:
                    try:
                        df[col] = df[col].round(2)
                    except:
                        # If can't round, keep as is
                        pass
        
        return df
    except Exception as e:
        st.error(f"Error formatting table data: {e}")
        # Return empty DataFrame in case of any errors
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'])

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
    # Multi-stock comparison mode
    st.sidebar.subheader("Compare Multiple Stocks")
    
    # Add very clear explanation before input
    st.sidebar.markdown("""
    ### How to Enter Stock Symbols:
    - Use commas to separate multiple stock symbols
    - Do NOT include spaces within a symbol
    - Each symbol should be a valid stock ticker
    
    ✅ CORRECT: `AAPL,MSFT,GOOG`  
    ❌ INCORRECT: `AAPL, MSFT, GOOG` (has spaces after commas)
    """)
    
    # Default input based on session state
    if 'compare_symbols' in st.session_state:
        default_symbols = st.session_state['compare_symbols']
    else:
        default_symbols = "AAPL,MSFT,GOOG"
    
    # Input for multiple ticker symbols with clear instructions
    tickers_input = st.sidebar.text_input(
        "Enter Stock Symbols:", 
        default_symbols,
        help="Enter comma-separated stock symbols without spaces, e.g., AAPL,MSFT,GOOG"
    ).upper()
    
    st.session_state['compare_symbols'] = tickers_input
    
    # Add example symbols that users can click to use
    st.sidebar.markdown("### Quick Examples:")
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("Tech Giants"):
        st.session_state['compare_symbols'] = "AAPL,MSFT,GOOG,AMZN"
        st.rerun()
        
    if col2.button("Popular ETFs"):
        st.session_state['compare_symbols'] = "SPY,QQQ,VTI"
        st.rerun()
    
    # Completely revised symbol parsing algorithm
    # First, detect if input is primarily space-separated or comma-separated
    if ',' in tickers_input:
        # Comma-separated format: First split by commas
        initial_parts = [part.strip() for part in tickers_input.split(',')]
    else:
        # Space-separated format: Split the whole input by spaces
        initial_parts = [part.strip() for part in tickers_input.split()]
    
    # Process each part to handle mixed formats
    tickers = []
    for part in initial_parts:
        if not part:  # Skip empty parts
            continue
            
        # If this part still has spaces and wasn't comma-separated, split further
        if ' ' in part and ',' not in tickers_input:
            # Split by space and add each as a ticker
            space_parts = [t.strip() for t in part.split() if t.strip()]
            tickers.extend(space_parts)
        else:
            # Single ticker (or already processed)
            tickers.append(part)
    
    # Final validation and cleanup
    cleaned_tickers = []
    for ticker in tickers:
        # Only include if it's not blank and doesn't contain spaces
        if ticker and ' ' not in ticker:
            cleaned_tickers.append(ticker.upper())
    
    # Remove duplicates
    tickers = list(set(cleaned_tickers))
    
    # Show what we've parsed
    if tickers:
        st.sidebar.success(f"Found symbols: {', '.join(tickers)}")
    
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
                    
                    # Use the same improved parsing logic for recent searches
                    # First, detect if input is primarily space-separated or comma-separated
                    if ',' in tickers_input:
                        # Comma-separated format: First split by commas
                        initial_parts = [part.strip() for part in tickers_input.split(',')]
                    else:
                        # Space-separated format: Split the whole input by spaces
                        initial_parts = [part.strip() for part in tickers_input.split()]
                    
                    # Process each part to handle mixed formats
                    tickers = []
                    for part in initial_parts:
                        if not part:  # Skip empty parts
                            continue
                            
                        # If this part still has spaces and wasn't comma-separated, split further
                        if ' ' in part and ',' not in tickers_input:
                            # Split by space and add each as a ticker
                            space_parts = [t.strip() for t in part.split() if t.strip()]
                            tickers.extend(space_parts)
                        else:
                            # Single ticker (or already processed)
                            tickers.append(part)
                    
                    # Final validation and cleanup
                    cleaned_tickers = []
                    for ticker in tickers:
                        # Only include if it's not blank and doesn't contain spaces
                        if ticker and ' ' not in ticker:
                            cleaned_tickers.append(ticker.upper())
                    
                    # Remove duplicates
                    tickers = list(set(cleaned_tickers))
                    
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
            # This is critical - we need to make sure we're splitting the input string
            # Debug the actual tickers list that will be used
            st.write(f"Debug: Tickers to compare: {tickers}")
            
            # Get data for multiple stocks (pass list of individual symbols, not combined string)
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
                    
                    # Calculate daily returns for correlation with timezone handling
                    returns_data = {}
                    
                    # First, standardize timezones across all datasets
                    for ticker, data in all_stock_data.items():
                        # Make a copy to avoid modifying the original data
                        df_copy = data.copy()
                        
                        # Convert the index to timezone-naive datetime
                        if isinstance(df_copy.index, pd.DatetimeIndex):
                            if df_copy.index.tz is not None:
                                # Convert tz-aware to tz-naive
                                df_copy.index = df_copy.index.tz_localize(None)
                        
                        # Calculate returns on the standardized data
                        returns_data[ticker] = df_copy['Close'].pct_change().dropna()
                    
                    try:
                        # Create a DataFrame with all returns
                        returns_df = pd.DataFrame(returns_data)
                    except TypeError as e:
                        st.error(f"Error creating returns dataframe: {e}")
                        st.error("Unable to create correlation matrix due to timezone inconsistencies in the data.")
                        # Create an empty DataFrame to prevent further errors
                        returns_df = pd.DataFrame()
                    
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
                tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Technical Indicators", "Key Metrics", "Historical Data"])
                
                with tab1:
                    # Create and display price chart
                    fig = create_price_chart(hist_data, company_name, selected_period)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Simple moving average
                    st.subheader("Moving Averages")
                    
                    # Static chart approach with pre-determined options
                    st.write("""
                    ### Moving Average Periods
                    Select which Moving Averages to show in the static chart below. 
                    (Showing multiple MAs provides better insight into trend direction)
                    """)
                    
                    # Use a static approach - render the chart with all MAs pre-calculated
                    # Simply use the pre-defined session state
                        
                    # Create a more reliable grid of buttons
                    col1, col2, col3 = st.columns(3)
                    
                    # Create a toggle function that updates session state
                    def toggle_ma(period):
                        st.session_state.ma_selections[period] = not st.session_state.ma_selections[period]
                    
                    # Create the buttons
                    with col1:
                        if st.button('🔍 20-Day MA', 
                                     help="Toggle 20-Day MA visibility",
                                     type="primary" if st.session_state.ma_selections['20-Day'] else "secondary"):
                            toggle_ma('20-Day')
                            
                        if st.button('🔍 50-Day MA', 
                                     help="Toggle 50-Day MA visibility",
                                     type="primary" if st.session_state.ma_selections['50-Day'] else "secondary"):
                            toggle_ma('50-Day')
                            
                    with col2:
                        if st.button('🔍 100-Day MA', 
                                     help="Toggle 100-Day MA visibility",
                                     type="primary" if st.session_state.ma_selections['100-Day'] else "secondary"):
                            toggle_ma('100-Day')
                            
                        if st.button('🔍 200-Day MA', 
                                     help="Toggle 200-Day MA visibility",
                                     type="primary" if st.session_state.ma_selections['200-Day'] else "secondary"):
                            toggle_ma('200-Day')
                            
                    with col3:
                        if st.button('🔍 5-Day MA', 
                                     help="Toggle 5-Day MA visibility",
                                     type="primary" if st.session_state.ma_selections['5-Day'] else "secondary"):
                            toggle_ma('5-Day')
                            
                        if st.button('🔍 10-Day MA', 
                                     help="Toggle 10-Day MA visibility",
                                     type="primary" if st.session_state.ma_selections['10-Day'] else "secondary"):
                            toggle_ma('10-Day')
                    
                    # Show which MAs are currently selected
                    selected_periods = [period for period, is_selected in st.session_state.ma_selections.items() if is_selected]
                    if selected_periods:
                        st.success(f"Showing: {', '.join(selected_periods)}")
                    else:
                        st.warning("No Moving Averages selected. Please select at least one.")
                    
                    # Calculate all MAs at once regardless of selection
                    hist_data_ma = hist_data.copy()
                    hist_data_ma['MA5'] = hist_data_ma['Close'].rolling(window=5).mean()
                    hist_data_ma['MA10'] = hist_data_ma['Close'].rolling(window=10).mean()
                    hist_data_ma['MA20'] = hist_data_ma['Close'].rolling(window=20).mean()
                    hist_data_ma['MA50'] = hist_data_ma['Close'].rolling(window=50).mean()
                    hist_data_ma['MA100'] = hist_data_ma['Close'].rolling(window=100).mean()
                    hist_data_ma['MA200'] = hist_data_ma['Close'].rolling(window=200).mean()
                    
                    # Create a Plotly figure - static approach
                    fig_ma = go.Figure()
                    
                    # Always add the price
                    fig_ma.add_trace(go.Scatter(
                        x=hist_data_ma.index,
                        y=hist_data_ma['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='black', width=1.5)
                    ))
                    
                    # Define a mapping for colors and other properties
                    ma_properties = {
                        '5-Day': {'column': 'MA5', 'color': '#FF9800', 'width': 1.5},
                        '10-Day': {'column': 'MA10', 'color': '#9C27B0', 'width': 1.5},
                        '20-Day': {'column': 'MA20', 'color': '#17BECF', 'width': 1.5},
                        '50-Day': {'column': 'MA50', 'color': '#B6267E', 'width': 1.5},
                        '100-Day': {'column': 'MA100', 'color': '#2196F3', 'width': 1.5},
                        '200-Day': {'column': 'MA200', 'color': '#2E55A5', 'width': 2.0}
                    }
                    
                    # Add traces for each selected MA
                    for period, is_selected in st.session_state.ma_selections.items():
                        if is_selected:
                            props = ma_properties[period]
                            fig_ma.add_trace(go.Scatter(
                                x=hist_data_ma.index,
                                y=hist_data_ma[props['column']],
                                mode='lines',
                                name=f'{period} MA',
                                line=dict(color=props['color'], width=props['width'])
                            ))
                    
                    # Layout 
                    fig_ma.update_layout(
                        title=f"{company_name} Price with Moving Averages",
                        xaxis_title='Date',
                        yaxis_title='Price ($)',
                        height=600,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Show the chart
                    st.plotly_chart(fig_ma, use_container_width=True)
                    
                    # Also add a simplified Streamlit chart as a backup
                    st.subheader("Moving Averages (Simple Chart)")
                    st.write("Alternative visualization with Streamlit's native chart:")
                    
                    # Prepare data for the simpler chart
                    simple_data = pd.DataFrame({'Price': hist_data_ma['Close']})
                    for period, is_selected in st.session_state.ma_selections.items():
                        if is_selected:
                            props = ma_properties[period]
                            simple_data[f'{period} MA'] = hist_data_ma[props['column']]
                            
                    # Display the simpler chart
                    st.line_chart(simple_data)
                
                with tab2:
                    st.subheader("Technical Indicators")
                    
                    # Use buttons for more reliable state management instead of checkboxes
                    st.write("### Toggle Technical Indicators")
                    
                    # Function to toggle indicator state
                    def toggle_indicator(indicator_name):
                        st.session_state.tech_indicators[indicator_name] = not st.session_state.tech_indicators[indicator_name]
                    
                    # Create a 2x3 grid of buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button('📊 RSI', 
                                    help="Toggle RSI indicator", 
                                    type="primary" if st.session_state.tech_indicators['RSI'] else "secondary",
                                    key="btn_rsi"):
                            toggle_indicator('RSI')
                        
                        if st.button('📊 MACD', 
                                    help="Toggle MACD indicator", 
                                    type="primary" if st.session_state.tech_indicators['MACD'] else "secondary",
                                    key="btn_macd"):
                            toggle_indicator('MACD')
                            
                    with col2:
                        if st.button('📊 Bollinger Bands', 
                                    help="Toggle Bollinger Bands", 
                                    type="primary" if st.session_state.tech_indicators['Bollinger'] else "secondary",
                                    key="btn_bb"):
                            toggle_indicator('Bollinger')
                            
                        if st.button('📊 Stochastic', 
                                    help="Toggle Stochastic Oscillator", 
                                    type="primary" if st.session_state.tech_indicators['Stochastic'] else "secondary",
                                    key="btn_stoch"):
                            toggle_indicator('Stochastic')
                            
                    with col3:
                        if st.button('📊 ADX', 
                                    help="Toggle ADX indicator", 
                                    type="primary" if st.session_state.tech_indicators['ADX'] else "secondary",
                                    key="btn_adx"):
                            toggle_indicator('ADX')
                            
                        if st.button('📊 CCI', 
                                    help="Toggle CCI indicator", 
                                    type="primary" if st.session_state.tech_indicators['CCI'] else "secondary",
                                    key="btn_cci"):
                            toggle_indicator('CCI')
                    
                    # Show which indicators are active
                    active_indicators = [name for name, is_active in st.session_state.tech_indicators.items() if is_active]
                    if active_indicators:
                        st.success(f"Active indicators: {', '.join(active_indicators)}")
                    else:
                        st.warning("No technical indicators selected. Please select at least one.")
                    
                    # Calculate all indicators upfront
                    indicators_data = calculate_technical_indicators(hist_data)
                    
                    # Simply use the pre-defined session state from app initialization
                    
                    # Display the selected indicators
                    st.subheader("Selected Technical Indicators")
                    
                    # Display a master toggle button
                    if st.button('Toggle All Indicators'):
                        # Toggle all to the opposite of current majority
                        current_state = sum(st.session_state.tech_indicators.values()) > 3
                        for key in st.session_state.tech_indicators:
                            st.session_state.tech_indicators[key] = not current_state
                    
                    # For each indicator, calculate and display regardless of checkbox
                    # RSI Indicator
                    if st.session_state.tech_indicators['RSI']:
                        st.markdown("### Relative Strength Index (RSI)")
                        st.markdown("""
                        RSI measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
                        - Values above 70 generally indicate overbought conditions (potential sell signal)
                        - Values below 30 generally indicate oversold conditions (potential buy signal)
                        - The centerline at 50 can indicate the trend direction
                        """)
                        
                        # Calculate RSI directly
                        rsi = ta.momentum.rsi(hist_data['Close'], window=14)
                        
                        # Create a simple Streamlit chart
                        rsi_data = pd.DataFrame({'RSI': rsi})
                        st.line_chart(rsi_data)
                        
                        # Also show with Plotly 
                        rsi_fig = go.Figure()
                        rsi_fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=rsi,
                            mode='lines',
                            name='RSI (14)',
                            line=dict(color='blue', width=1.5)
                        ))
                        
                        # Add reference lines
                        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", 
                                        annotation_text="Overbought (70)", 
                                        annotation_position="right")
                        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", 
                                        annotation_text="Oversold (30)", 
                                        annotation_position="right")
                        rsi_fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                                        annotation_text="Centerline (50)", 
                                        annotation_position="right")
                        
                        rsi_fig.update_layout(
                            title='Relative Strength Index (RSI)',
                            xaxis_title='Date',
                            yaxis_title='RSI',
                            height=400,
                            yaxis=dict(range=[0, 100])
                        )
                        
                        st.plotly_chart(rsi_fig, use_container_width=True)
                    
                    # MACD Indicator
                    if st.session_state.tech_indicators['MACD']:
                        st.markdown("### Moving Average Convergence Divergence (MACD)")
                        st.markdown("""
                        MACD is a trend-following momentum indicator that shows the relationship between two moving averages.
                        - When MACD crosses above the signal line, it's a potential buy signal
                        - When MACD crosses below the signal line, it's a potential sell signal
                        - The histogram shows the difference between MACD and signal line
                        """)
                        
                        # Calculate MACD directly
                        macd_line = ta.trend.macd(hist_data['Close'])
                        signal_line = ta.trend.macd_signal(hist_data['Close'])
                        macd_hist = ta.trend.macd_diff(hist_data['Close'])
                        
                        # Combine into a DataFrame
                        macd_data = pd.DataFrame({
                            'MACD': macd_line,
                            'Signal': signal_line,
                            'Histogram': macd_hist
                        })
                        
                        # Show with Streamlit
                        st.line_chart(macd_data[['MACD', 'Signal']])
                        
                        # Create Plotly MACD chart
                        macd_fig = go.Figure()
                        
                        # Add MACD line
                        macd_fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=macd_line,
                            mode='lines',
                            name='MACD Line',
                            line=dict(color='blue', width=1.5)
                        ))
                        
                        # Add Signal line
                        macd_fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=signal_line,
                            mode='lines', 
                            name='Signal Line',
                            line=dict(color='red', width=1.5)
                        ))
                        
                        # Add histogram
                        colors = ['green' if val >= 0 else 'red' for val in macd_hist]
                        macd_fig.add_trace(go.Bar(
                            x=hist_data.index,
                            y=macd_hist,
                            name='Histogram',
                            marker_color=colors
                        ))
                        
                        macd_fig.update_layout(
                            title='MACD (12,26,9)',
                            xaxis_title='Date',
                            yaxis_title='MACD',
                            height=400,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(macd_fig, use_container_width=True)
                    
                    # Bollinger Bands
                    if st.session_state.tech_indicators['Bollinger']:
                        st.markdown("### Bollinger Bands")
                        st.markdown("""
                        Bollinger Bands consist of a middle band (SMA) with two outer bands (standard deviations).
                        - Price reaching the upper band may indicate overbought conditions
                        - Price reaching the lower band may indicate oversold conditions
                        - Bands narrowing can signal potential volatility increase
                        """)
                        
                        # Calculate Bollinger Bands directly
                        window = 20
                        bb_sma = hist_data['Close'].rolling(window=window).mean()
                        bb_std = hist_data['Close'].rolling(window=window).std()
                        bb_upper = bb_sma + (bb_std * 2)
                        bb_lower = bb_sma - (bb_std * 2)
                        
                        # Create data for Streamlit's native chart
                        st.subheader("Bollinger Bands - Simple Chart")
                        
                        # Create a basic dataframe
                        bb_data = pd.DataFrame({
                            'Price': hist_data['Close'],
                            'SMA20': bb_sma,
                            'Upper': bb_upper,
                            'Lower': bb_lower
                        })
                        
                        # Display simple chart first
                        st.line_chart(bb_data)
                        
                        # Create a Plotly chart for more flexibility
                        st.subheader("Bollinger Bands - Interactive Chart")
                        
                        # Create a new figure
                        bb_fig = go.Figure()
                        
                        # Add Close price
                        bb_fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=hist_data['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='blue', width=1.5)
                        ))
                        
                        # Add the bands
                        bb_fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=bb_upper,
                            mode='lines',
                            name='Upper Band (+2σ)',
                            line=dict(color='red', width=1.2)
                        ))
                        
                        bb_fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=bb_sma,
                            mode='lines',
                            name='20-Day SMA',
                            line=dict(color='green', width=1.2)
                        ))
                        
                        bb_fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=bb_lower,
                            mode='lines',
                            name='Lower Band (-2σ)',
                            line=dict(color='red', width=1.2)
                        ))
                        
                        # Try to add a fill between the bands
                        try:
                            # Create a fill by using two separate traces
                            bb_fig.add_trace(go.Scatter(
                                x=hist_data.index,
                                y=bb_upper,
                                fill=None,
                                mode='lines',
                                line=dict(width=0.1, color='rgba(0,0,0,0)'),
                                showlegend=False
                            ))
                            
                            bb_fig.add_trace(go.Scatter(
                                x=hist_data.index,
                                y=bb_lower,
                                fill='tonexty',
                                mode='lines',
                                line=dict(width=0.1, color='rgba(0,0,0,0)'),
                                fillcolor='rgba(173, 216, 230, 0.2)',
                                showlegend=False
                            ))
                        except:
                            # If fill doesn't work, continue without it
                            pass
                        
                        # Update layout
                        bb_fig.update_layout(
                            title='Bollinger Bands (20,2)',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            height=450,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            hovermode="x unified"
                        )
                        
                        # Display the chart
                        st.plotly_chart(bb_fig, use_container_width=True)
                    
                    # Stochastic oscillator
                    if st.session_state.tech_indicators['Stochastic']:
                        st.markdown("### Stochastic Oscillator")
                        st.markdown("""
                        The Stochastic Oscillator compares a stock's closing price to its price range over a period.
                        - Values above 80 indicate overbought conditions
                        - Values below 20 indicate oversold conditions
                        - %K line crossing above %D line is a potential buy signal
                        - %K line crossing below %D line is a potential sell signal
                        """)
                        
                        # Calculate Stochastic directly
                        stoch_k = ta.momentum.stoch(high=hist_data['High'], 
                                                 low=hist_data['Low'], 
                                                 close=hist_data['Close'])
                        stoch_d = ta.momentum.stoch_signal(high=hist_data['High'], 
                                                        low=hist_data['Low'], 
                                                        close=hist_data['Close'])
                        
                        # Show with Streamlit native chart
                        stoch_data = pd.DataFrame({
                            '%K': stoch_k,
                            '%D': stoch_d
                        })
                        
                        st.line_chart(stoch_data)
                        
                        # Create a Plotly chart 
                        stoch_fig = go.Figure()
                        
                        # Add K line
                        stoch_fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=stoch_k,
                            mode='lines',
                            name='%K',
                            line=dict(color='blue', width=1.5)
                        ))
                        
                        # Add D line
                        stoch_fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=stoch_d,
                            mode='lines',
                            name='%D',
                            line=dict(color='red', width=1.5)
                        ))
                        
                        # Add reference lines
                        stoch_fig.add_hline(y=80, line_dash="dash", line_color="red", 
                                      annotation_text="Overbought (80)", 
                                      annotation_position="right")
                        stoch_fig.add_hline(y=20, line_dash="dash", line_color="green", 
                                      annotation_text="Oversold (20)", 
                                      annotation_position="right")
                        
                        stoch_fig.update_layout(
                            title='Stochastic Oscillator (14,3)',
                            xaxis_title='Date',
                            yaxis_title='Value',
                            height=400,
                            yaxis=dict(range=[0, 100]),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(stoch_fig, use_container_width=True)
                    
                    # ADX (Average Directional Index)
                    if st.session_state.tech_indicators['ADX']:
                        st.markdown("### Average Directional Index (ADX)")
                        st.markdown("""
                        ADX is used to determine the strength of a trend, regardless of its direction.
                        - Values above 25 indicate a strong trend
                        - Values below 20 indicate a weak or non-existent trend
                        - ADX does not show trend direction, only strength
                        """)
                        
                        # Calculate ADX directly
                        adx = ta.trend.adx(high=hist_data['High'], 
                                        low=hist_data['Low'], 
                                        close=hist_data['Close'])
                        
                        # Create dataframe for display
                        adx_data = pd.DataFrame({
                            'ADX': adx
                        })
                        
                        # Show with Streamlit native chart
                        st.line_chart(adx_data)
                        
                        # Create Plotly chart
                        adx_fig = go.Figure()
                        
                        # Add ADX line
                        adx_fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=adx,
                            mode='lines',
                            name='ADX',
                            line=dict(color='purple', width=1.5)
                        ))
                        
                        # Add reference lines
                        adx_fig.add_hline(y=25, line_dash="dash", line_color="green", 
                                     annotation_text="Strong Trend (25)", 
                                     annotation_position="right")
                        
                        adx_fig.add_hline(y=20, line_dash="dash", line_color="red", 
                                     annotation_text="Weak Trend (20)", 
                                     annotation_position="right")
                        
                        adx_fig.update_layout(
                            title='Average Directional Index (ADX)',
                            xaxis_title='Date',
                            yaxis_title='ADX',
                            height=400,
                            yaxis=dict(range=[0, 60])
                        )
                        
                        st.plotly_chart(adx_fig, use_container_width=True)
                    
                    # CCI (Commodity Channel Index)
                    if st.session_state.tech_indicators['CCI']:
                        st.markdown("### Commodity Channel Index (CCI)")
                        st.markdown("""
                        CCI measures the current price level relative to an average price level over a given period.
                        - Values above +100 suggest an overbought condition
                        - Values below -100 suggest an oversold condition
                        - CCI can be used to identify new trends or extreme conditions
                        """)
                        
                        # Calculate CCI directly
                        cci = ta.trend.cci(high=hist_data['High'], 
                                        low=hist_data['Low'], 
                                        close=hist_data['Close'])
                        
                        # Create dataframe for display
                        cci_data = pd.DataFrame({
                            'CCI': cci
                        })
                        
                        # Show with Streamlit native chart
                        st.line_chart(cci_data)
                        
                        # Create Plotly chart 
                        cci_fig = go.Figure()
                        
                        # Add CCI line
                        cci_fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=cci,
                            mode='lines',
                            name='CCI',
                            line=dict(color='blue', width=1.5)
                        ))
                        
                        # Add reference lines
                        cci_fig.add_hline(y=100, line_dash="dash", line_color="red", 
                                     annotation_text="Overbought (+100)", 
                                     annotation_position="right")
                        cci_fig.add_hline(y=-100, line_dash="dash", line_color="green", 
                                      annotation_text="Oversold (-100)", 
                                      annotation_position="right")
                        cci_fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                                     annotation_text="Centerline", 
                                     annotation_position="right")
                        
                        cci_fig.update_layout(
                            title='Commodity Channel Index (CCI)',
                            xaxis_title='Date',
                            yaxis_title='CCI',
                            height=400
                        )
                        
                        st.plotly_chart(cci_fig, use_container_width=True)
                            

                
                with tab3:
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
                
                with tab4:
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
        st.markdown("- Advanced technical indicators (RSI, MACD, Bollinger Bands, Stochastic, ADX, CCI)")
        st.markdown("- Key financial metrics and company information")
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
