import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

def main():
    st.title("Basic Stock Price Chart")
    st.markdown("This simple app always shows a stock price chart.")
    
    # Simple input for stock symbol
    ticker_symbol = st.text_input("Enter a stock symbol (e.g., AAPL):", "AAPL")
    period = st.selectbox("Select time period:", 
                         ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], 
                         index=3)
    
    if st.button("Show Chart"):
        # Fetch data
        stock_data = yf.download(ticker_symbol, period=period)
        
        if not stock_data.empty:
            # Create price chart
            fig = go.Figure()
            
            # Add price trace
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{ticker_symbol} Stock Price - {period}",
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500,
                hovermode="x unified"
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display data table
            st.subheader("Stock Data")
            st.dataframe(stock_data)
        else:
            st.error(f"Could not retrieve data for {ticker_symbol}")

if __name__ == "__main__":
    main()