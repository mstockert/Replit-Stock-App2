import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import database as db
from sqlalchemy.exc import OperationalError, SQLAlchemyError

def handle_db_error(func):
    """Decorator for handling database errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (OperationalError, SQLAlchemyError) as e:
            st.error(f"Database Error: {e}")
            return None
    return wrapper

def create_default_watchlist():
    """Create a default watchlist if none exists"""
    try:
        watchlists = db.get_all_watchlists()
        if not watchlists:
            watchlist_id = db.create_watchlist("My Watchlist", "Default watchlist")
            if watchlist_id:
                # Add some popular stocks
                default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
                for symbol in default_stocks:
                    db.add_stock_to_watchlist(watchlist_id, symbol)
                st.success("Created default watchlist with popular stocks")
            else:
                st.warning("Could not create default watchlist")
    except Exception as e:
        st.warning(f"Error creating default watchlist: {e}")
        # Return a default watchlist
        return [{'id': 1, 'name': 'Default Watchlist', 'description': 'Generated when database is unavailable'}]

def display_watchlist_selector():
    """Display a selector for watchlists"""
    try:
        watchlists = db.get_all_watchlists()
        
        # If no watchlists, create default
        if not watchlists:
            create_default_watchlist()
            watchlists = db.get_all_watchlists()
            
            # If still no watchlists, show a meaningful error
            if not watchlists:
                st.error("Could not create or access watchlists. Database may be unavailable.")
                return None
        
        # Display watchlist selector
        watchlist_names = [w['name'] for w in watchlists]
        selected_watchlist_name = st.selectbox("Select Watchlist", watchlist_names)
        
        # Get the selected watchlist ID
        selected_watchlist = next((w for w in watchlists if w['name'] == selected_watchlist_name), None)
        
        return selected_watchlist
    except Exception as e:
        st.error(f"Error accessing watchlists: {e}")
        return None

def display_watchlist_manager():
    """Display a UI for managing watchlists"""
    try:
        st.header("Watchlist Manager")
        
        # Create two columns
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Display watchlist selector
            selected_watchlist = display_watchlist_selector()
            
            if selected_watchlist:
                try:
                    # Display stocks in the watchlist
                    stocks = db.get_watchlist_stocks(selected_watchlist['id'])
                    
                    if stocks:
                        st.subheader(f"Stocks in {selected_watchlist['name']}")
                        
                        # Create a DataFrame for display
                        stocks_df = pd.DataFrame(stocks)
                        st.dataframe(stocks_df[['symbol', 'name']], hide_index=True)
                        
                        # Option to analyze watchlist
                        if st.button("Compare Watchlist Stocks"):
                            # Store the symbols in session state for comparison
                            symbols = [stock['symbol'] for stock in stocks]
                            st.session_state['compare_symbols'] = ','.join(symbols)
                            st.session_state['analysis_type'] = "Stock Comparison"
                            st.rerun()
                    else:
                        st.info("No stocks in this watchlist. Add some stocks using the form on the right.")
                except Exception as e:
                    st.error(f"Error retrieving watchlist stocks: {e}")
        
        with col2:
            st.subheader("Add to Watchlist")
            
            # Form for adding a stock to the watchlist
            with st.form("add_to_watchlist_form"):
                stock_symbol = st.text_input("Stock Symbol").upper()
                submit_button = st.form_submit_button("Add to Watchlist")
                
                if submit_button and stock_symbol and selected_watchlist:
                    try:
                        db.add_stock_to_watchlist(selected_watchlist['id'], stock_symbol)
                        st.success(f"Added {stock_symbol} to {selected_watchlist['name']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding stock to watchlist: {e}")
            
            # Form for creating a new watchlist
            st.subheader("Create New Watchlist")
            with st.form("create_watchlist_form"):
                watchlist_name = st.text_input("Watchlist Name")
                watchlist_desc = st.text_area("Description (optional)")
                create_button = st.form_submit_button("Create Watchlist")
                
                if create_button and watchlist_name:
                    try:
                        watchlist_id = db.create_watchlist(watchlist_name, watchlist_desc)
                        if watchlist_id:
                            st.success(f"Created new watchlist: {watchlist_name}")
                            st.rerun()
                        else:
                            st.error("Could not create watchlist. Database may be unavailable.")
                    except Exception as e:
                        st.error(f"Error creating watchlist: {e}")
    except Exception as e:
        st.error(f"Error displaying watchlist manager: {e}")
        st.info("Database may be temporarily unavailable. Please try again later.")

def save_search_to_history(query, search_type):
    """Save a search query to history"""
    try:
        db.add_search_to_history(query, search_type)
    except Exception as e:
        print(f"Error saving search history: {e}")
        # Continue without saving search history

def display_recent_searches():
    """Display recent search history"""
    try:
        searches = db.get_search_history(limit=5)
        
        if searches:
            st.subheader("Recent Searches")
            
            # Display as clickable buttons
            for search in searches:
                query = search['query']
                search_type = search['type']
                
                # For comparison searches, split the query
                if ',' in query and search_type == 'comparison':
                    button_label = f"Compare: {query}"
                else:
                    button_label = query
                    
                if st.button(button_label):
                    if search_type == 'comparison':
                        st.session_state['compare_symbols'] = query
                        st.session_state['analysis_type'] = "Stock Comparison"
                    else:
                        st.session_state['ticker'] = query
                        st.session_state['analysis_type'] = "Single Stock Analysis"
                    st.rerun()
    except Exception as e:
        st.info("Search history is currently unavailable")
        print(f"Error displaying search history: {e}")

def cache_stock_data(ticker_symbol, hist_data):
    """Cache stock data in the database"""
    try:
        # Add the stock to the database if it doesn't exist
        stock_id = db.add_stock(ticker_symbol)
        
        if stock_id is None:
            print("Could not add stock to database, skipping cache operation")
            return
        
        # Format the data for database storage
        if not hist_data.empty:
            # Reset index to make Date a column
            if isinstance(hist_data.index, pd.DatetimeIndex):
                price_data = hist_data.reset_index()
            else:
                price_data = hist_data.copy()
                
            # Add to database
            db.add_stock_prices(stock_id, price_data)
    except Exception as e:
        print(f"Error caching stock data: {e}")
        # Continue without caching data