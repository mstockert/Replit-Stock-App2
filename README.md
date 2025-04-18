# Stock Market Dashboard

A Streamlit-based stock data visualization platform that provides comprehensive financial insights through interactive and user-friendly interfaces.

## Features

- Stock data visualization with interactive charts
- Multi-stock comparison with correlation analysis
- One-click example comparisons for common stock groupings
- Stock price history tables and trend visualization
- Watchlist management for tracking favorite stocks
- Data export functionality (CSV)
- Database storage for persistent data
- Robust error handling and database connectivity fallbacks
- Timezone consistency handling for mixed data sources

## Technology Stack

- **Frontend**: Streamlit
- **Data Source**: Yahoo Finance API (yfinance)
- **Visualization**: Plotly
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Data Processing**: Pandas, NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/mstockert/Replit-Stock-App2.git
   cd Replit-Stock-App2
   ```

2. Set up a virtual environment (recommended):
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

   If requirements.txt is missing, install these packages:
   ```
   pip install streamlit pandas numpy plotly yfinance sqlalchemy psycopg2-binary
   ```

4. Set up PostgreSQL database:
   
   **Option 1: Using PostgreSQL**
   - Install PostgreSQL if not already installed
   - Create a database for the application
   - Set environment variable: `DATABASE_URL="postgresql://user:password@host/dbname"`
   
   On macOS with Homebrew:
   ```bash
   brew install postgresql
   brew services start postgresql
   createdb stockapp
   export DATABASE_URL="postgresql://localhost:5432/stockapp"
   ```

   **Option 2: Using SQLite (fallback)**
   - The application will automatically use an in-memory SQLite database if PostgreSQL is not available

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your browser to http://localhost:8501

3. Enter stock symbols (e.g., AAPL, MSFT, GOOG) to visualize data

4. Create watchlists to track your favorite stocks

## Project Structure

- `app.py`: Main Streamlit application with UI components and data visualization
- `database.py`: Database models and functions for data persistence
- `db_ui.py`: UI components for database operations (watchlists, search history)
- `.streamlit/config.toml`: Streamlit configuration

## Error Handling

The application includes robust error handling for:
- Database connection issues
- API failures
- Invalid stock symbols
- Data retrieval errors
- Timezone inconsistencies between data sources

## Recent Improvements

### Stock Comparison Enhancement (April 2025)
- Fixed parsing and validation of multiple stock symbols
- Added one-click example buttons for quick comparisons
- Improved user interface with clearer instructions
- Fixed timezone inconsistency issues when combining data
- Enhanced error handling and debugging information

## License

This project is for personal use only.