# Stock Market Dashboard

A Streamlit-based stock data visualization platform that provides comprehensive financial insights through interactive and user-friendly interfaces.

## Features

- Stock data visualization with interactive charts
- Multi-stock comparison with correlation analysis
- Stock price history tables and trend visualization
- Watchlist management for tracking favorite stocks
- Data export functionality (CSV)
- Database storage for persistent data

## Technology Stack

- **Frontend**: Streamlit
- **Data Source**: Yahoo Finance API (yfinance)
- **Visualization**: Plotly
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Data Processing**: Pandas, NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/mstockert/Replit-Stock-App.git
   cd Replit-Stock-App
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up PostgreSQL database (optional):
   - Create a database for the application
   - Set environment variable: `DATABASE_URL="postgresql://user:password@host/dbname"`

## Usage

Run the application with:
```
streamlit run app.py
```

Then open your browser to http://localhost:8501

## Project Structure

- `app.py`: Main Streamlit application
- `database.py`: Database models and functions
- `db_ui.py`: UI components for database operations
- `.streamlit/config.toml`: Streamlit configuration

## License

This project is for personal use only.