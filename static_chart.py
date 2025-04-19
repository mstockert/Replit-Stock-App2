import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Static Chart Demo", layout="wide")

# App title
st.title("Static Chart Demo")
st.write("This is the simplest possible chart implementation.")

# Get Apple data - no user input required
try:
    data = yf.download("AAPL", period="1y")
    
    # Create the simplest possible chart
    st.subheader("Apple Stock Price (1 Year)")
    
    # Use matplotlib for maximum simplicity
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot price
    ax.plot(data.index, data['Close'], label='AAPL Close Price')
    
    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Display the chart
    st.pyplot(fig)
    
    # Also show a basic Streamlit native chart
    st.subheader("Streamlit Native Chart")
    st.line_chart(data['Close'])
    
    # Show some data
    st.subheader("Stock Data")
    st.dataframe(data.head(10))
    
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Detailed error information for debugging:")
    st.exception(e)