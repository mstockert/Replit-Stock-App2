import os
import pandas as pd
from datetime import datetime
import time
import streamlit as st
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy import text as sql_text  # Renamed to avoid confusion
from sqlalchemy import Text as SQLText  # Renamed to avoid confusion
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import OperationalError, SQLAlchemyError

# Get the database URL from environment variables
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create a function to get a database engine with retry mechanism
def get_engine(max_retries=3, retry_delay=2):
    """Get a SQLAlchemy engine with retry mechanism"""
    retry_count = 0
    last_error = None

    # If no DATABASE_URL is provided, use in-memory SQLite
    if not DATABASE_URL:
        print("No DATABASE_URL provided, using in-memory SQLite database")
        return create_engine('sqlite:///:memory:')

    while retry_count < max_retries:
        try:
            # Create SQLAlchemy engine with connection pooling and ping
            engine = create_engine(
                DATABASE_URL, 
                pool_pre_ping=True,   # Test connection before use
                pool_recycle=3600,    # Recycle connections after 1 hour
                connect_args={
                    "connect_timeout": 10  # Timeout after 10 seconds
                }
            )
            # Test connection with simple query
            with engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
            return engine
        except (OperationalError, SQLAlchemyError) as e:
            retry_count += 1
            last_error = e
            print(f"Database connection error (attempt {retry_count}/{max_retries}): {e}")
            time.sleep(retry_delay)
    
    print(f"Failed to connect to database after {max_retries} attempts: {last_error}")
    print("Falling back to in-memory SQLite database")
    # Fallback to in-memory SQLite database if PostgreSQL is unavailable
    return create_engine('sqlite:///:memory:')

# Create SQLAlchemy engine with retry and fallback
engine = get_engine()
print(f"Using database engine: {engine.name}")

# Create declarative base
Base = declarative_base()

# Define the models
class Stock(Base):
    """Stock table to store stock symbols and names"""
    __tablename__ = 'stocks'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False)
    name = Column(String(100))
    sector = Column(String(100))
    industry = Column(String(100))
    date_added = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    price_data = relationship("StockPrice", back_populates="stock", cascade="all, delete-orphan")
    watchlists = relationship("WatchlistStock", back_populates="stock")
    
    def __repr__(self):
        return f"<Stock(symbol='{self.symbol}', name='{self.name}')>"

class StockPrice(Base):
    """Stock price history"""
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    dividends = Column(Float)
    stock_splits = Column(Float)
    
    # Relationship
    stock = relationship("Stock", back_populates="price_data")
    
    def __repr__(self):
        return f"<StockPrice(stock_id={self.stock_id}, date='{self.date}', close={self.close})>"

class Watchlist(Base):
    """User watchlists"""
    __tablename__ = 'watchlists'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500))  # Using String instead of Text
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    stocks = relationship("WatchlistStock", back_populates="watchlist", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Watchlist(name='{self.name}')>"

class WatchlistStock(Base):
    """Association table for watchlists and stocks"""
    __tablename__ = 'watchlist_stocks'
    
    id = Column(Integer, primary_key=True)
    watchlist_id = Column(Integer, ForeignKey('watchlists.id'), nullable=False)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    watchlist = relationship("Watchlist", back_populates="stocks")
    stock = relationship("Stock", back_populates="watchlists")
    
    def __repr__(self):
        return f"<WatchlistStock(watchlist_id={self.watchlist_id}, stock_id={self.stock_id})>"

class SearchHistory(Base):
    """Store user search history"""
    __tablename__ = 'search_history'
    
    id = Column(Integer, primary_key=True)
    query = Column(String(100), nullable=False)
    search_type = Column(String(50))  # 'single', 'comparison'
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SearchHistory(query='{self.query}', timestamp='{self.timestamp}')>"

# Create all tables
Base.metadata.create_all(engine)

# Create a session factory
Session = sessionmaker(bind=engine)

# Functions for database operations with error handling
def add_stock(symbol, name=None, sector=None, industry=None):
    """Add a stock to the database if it doesn't already exist"""
    session = None
    try:
        # Ensure engine exists
        if not engine:
            print("Database not available")
            return None
        
        # Create a session
        session = Session()
        
        # Check if stock already exists
        stock = session.query(Stock).filter(Stock.symbol == symbol).first()
        
        if not stock:
            # Create new stock
            stock = Stock(symbol=symbol, name=name, sector=sector, industry=industry)
            session.add(stock)
            session.commit()
            print(f"Added new stock: {symbol}")
        else:
            # Update stock info if provided
            if name and not stock.name:
                stock.name = name
            if sector and not stock.sector:
                stock.sector = sector
            if industry and not stock.industry:
                stock.industry = industry
            session.commit()
            print(f"Updated stock: {symbol}")
        
        stock_id = stock.id
        return stock_id
    except (OperationalError, SQLAlchemyError) as e:
        if session:
            session.rollback()
        print(f"Database error in add_stock: {e}")
        return None
    finally:
        if session:
            session.close()

def add_stock_prices(stock_id, price_data):
    """Add historical price data for a stock"""
    if price_data.empty:
        print("No price data to add")
        return 0
    
    session = None
    try:
        # Ensure engine exists
        if not engine:
            print("Database not available")
            return 0
        
        session = Session()
        
        # Reset index to make date a column
        if isinstance(price_data.index, pd.DatetimeIndex):
            price_data = price_data.reset_index()
        
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(price_data['Date']):
            price_data['Date'] = pd.to_datetime(price_data['Date'])
        
        # Check for existing data to avoid duplicates
        existing_dates = session.query(StockPrice.date)\
                            .filter(StockPrice.stock_id == stock_id)\
                            .all()
        existing_dates = [date[0].date() for date in existing_dates]
        
        # Add each row to the database
        prices_added = 0
        for _, row in price_data.iterrows():
            # Skip if date already exists
            if row['Date'].date() in existing_dates:
                continue
                
            # Create new price entry
            price = StockPrice(
                stock_id=stock_id,
                date=row['Date'],
                open=row.get('Open'),
                high=row.get('High'),
                low=row.get('Low'),
                close=row.get('Close'),
                volume=row.get('Volume'),
                dividends=row.get('Dividends', 0),
                stock_splits=row.get('Stock Splits', 0)
            )
            session.add(price)
            prices_added += 1
        
        session.commit()
        print(f"Added {prices_added} new price entries")
        return prices_added
    except (OperationalError, SQLAlchemyError) as e:
        if session:
            session.rollback()
        print(f"Database error in add_stock_prices: {e}")
        return 0
    finally:
        if session:
            session.close()

def get_stock_prices(symbol, start_date=None, end_date=None):
    """Get price data for a stock from the database"""
    session = None
    try:
        # Ensure engine exists
        if not engine:
            print("Database not available")
            return None
        
        session = Session()
        
        # Get stock ID
        stock = session.query(Stock).filter(Stock.symbol == symbol).first()
        if not stock:
            return None
            
        # Build query
        query = session.query(StockPrice).filter(StockPrice.stock_id == stock.id)
        
        # Add date filters if provided
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            query = query.filter(StockPrice.date >= start_date)
        
        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            query = query.filter(StockPrice.date <= end_date)
        
        # Order by date and fetch results
        prices = query.order_by(StockPrice.date).all()
        
        # Convert to DataFrame
        if prices:
            data = {
                'Date': [p.date for p in prices],
                'Open': [p.open for p in prices],
                'High': [p.high for p in prices],
                'Low': [p.low for p in prices],
                'Close': [p.close for p in prices],
                'Volume': [p.volume for p in prices],
                'Dividends': [p.dividends for p in prices],
                'Stock Splits': [p.stock_splits for p in prices]
            }
            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)
            return df
        
        return None
    except (OperationalError, SQLAlchemyError) as e:
        print(f"Database error in get_stock_prices: {e}")
        return None
    finally:
        if session:
            session.close()

def create_watchlist(name, description=None):
    """Create a new watchlist"""
    session = None
    try:
        # Ensure engine exists
        if not engine:
            print("Database not available")
            return None
        
        session = Session()
        
        # Check if watchlist already exists
        watchlist = session.query(Watchlist).filter(Watchlist.name == name).first()
        
        if not watchlist:
            # Create new watchlist
            watchlist = Watchlist(name=name, description=description)
            session.add(watchlist)
            session.commit()
            print(f"Created new watchlist: {name}")
            watchlist_id = watchlist.id
        else:
            print(f"Watchlist already exists: {name}")
            watchlist_id = watchlist.id
        
        return watchlist_id
    except (OperationalError, SQLAlchemyError) as e:
        if session:
            session.rollback()
        print(f"Database error in create_watchlist: {e}")
        return None
    finally:
        if session:
            session.close()

def add_stock_to_watchlist(watchlist_id, stock_symbol):
    """Add a stock to a watchlist"""
    session = None
    try:
        # Ensure engine exists
        if not engine:
            print("Database not available")
            return
        
        # Get stock ID (create if doesn't exist)
        session = Session()
        stock = session.query(Stock).filter(Stock.symbol == stock_symbol).first()
        
        if not stock:
            # We need to add the stock first
            session.close()
            session = None
            stock_id = add_stock(stock_symbol)
            if not stock_id:
                return  # Error adding stock
        else:
            stock_id = stock.id
            session.close()
            session = None
        
        # Check if stock is already in the watchlist
        session = Session()
        existing = session.query(WatchlistStock).filter(
            WatchlistStock.watchlist_id == watchlist_id,
            WatchlistStock.stock_id == stock_id
        ).first()
        
        if not existing:
            # Add stock to watchlist
            watchlist_stock = WatchlistStock(watchlist_id=watchlist_id, stock_id=stock_id)
            session.add(watchlist_stock)
            session.commit()
            print(f"Added {stock_symbol} to watchlist {watchlist_id}")
        else:
            print(f"{stock_symbol} is already in watchlist {watchlist_id}")
    except (OperationalError, SQLAlchemyError) as e:
        if session:
            session.rollback()
        print(f"Database error in add_stock_to_watchlist: {e}")
    finally:
        if session:
            session.close()

def get_watchlist_stocks(watchlist_id):
    """Get all stocks in a watchlist"""
    session = None
    try:
        # Ensure engine exists
        if not engine:
            print("Database not available")
            return []
        
        session = Session()
        
        # Join WatchlistStock with Stock to get stock info
        stocks = session.query(Stock)\
            .join(WatchlistStock, WatchlistStock.stock_id == Stock.id)\
            .filter(WatchlistStock.watchlist_id == watchlist_id)\
            .all()
        
        result = [{'id': s.id, 'symbol': s.symbol, 'name': s.name} for s in stocks]
        
        return result
    except (OperationalError, SQLAlchemyError) as e:
        print(f"Database error in get_watchlist_stocks: {e}")
        return []
    finally:
        if session:
            session.close()

def get_all_watchlists():
    """Get a list of all watchlists"""
    session = None
    try:
        # Ensure engine exists
        if not engine:
            print("Database not available")
            return []
        
        session = Session()
        
        # Query with extended timeout
        watchlists = session.query(Watchlist).all()
        result = [{'id': w.id, 'name': w.name, 'description': w.description} for w in watchlists]
        
        return result
    except (OperationalError, SQLAlchemyError) as e:
        print(f"Database error in get_all_watchlists: {e}")
        # Create a default watchlist if database is not available
        return [{'id': 1, 'name': 'Default Watchlist', 'description': 'Generated when database is unavailable'}]
    finally:
        if session:
            session.close()

def add_search_to_history(query, search_type='single'):
    """Add a search query to the history"""
    session = None
    try:
        # Ensure engine exists
        if not engine:
            print("Database not available")
            return
        
        session = Session()
        
        # Create new search history entry
        search = SearchHistory(query=query, search_type=search_type)
        session.add(search)
        session.commit()
    except (OperationalError, SQLAlchemyError) as e:
        if session:
            session.rollback()
        print(f"Database error in add_search_to_history: {e}")
    finally:
        if session:
            session.close()

def get_search_history(limit=10):
    """Get recent search history"""
    session = None
    try:
        # Ensure engine exists
        if not engine:
            print("Database not available")
            return []
        
        session = Session()
        
        searches = session.query(SearchHistory)\
            .order_by(SearchHistory.timestamp.desc())\
            .limit(limit)\
            .all()
        
        result = [{'id': s.id, 'query': s.query, 'type': s.search_type, 'timestamp': s.timestamp} for s in searches]
        
        return result
    except (OperationalError, SQLAlchemyError) as e:
        print(f"Database error in get_search_history: {e}")
        return []
    finally:
        if session:
            session.close()