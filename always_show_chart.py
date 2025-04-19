import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Always Show Chart Demo", layout="wide")

# App title
st.title("Always-Visible Chart Demo")
st.write("This app demonstrates a minimal approach to ensure charts always remain visible.")

# Default ticker
default_ticker = "AAPL"

# Function to get stock data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(ticker, period="1y"):
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            st.error(f"No data found for {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Get the stock data upfront (always load AAPL)
data = get_stock_data(default_ticker)

if data is not None:
    # Create a container for the chart that will ALWAYS be displayed
    chart_container = st.container()
    
    # Display checkboxes for MAs in a separate container
    st.subheader("Select Moving Averages")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_20ma = st.checkbox("20-Day MA", value=True)
    with col2:
        show_50ma = st.checkbox("50-Day MA", value=True)
    with col3:
        show_100ma = st.checkbox("100-Day MA", value=False)
    with col4:
        show_200ma = st.checkbox("200-Day MA", value=True)
    
    # Calculate MAs if needed
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA100'] = data['Close'].rolling(window=100).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # Now ALWAYS display a chart in the container, regardless of checkboxes
    with chart_container:
        st.subheader(f"{default_ticker} Price Chart")
        
        # Create figure
        fig = go.Figure()
        
        # ALWAYS add price data - this is the key part
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='black', width=2)
        ))
        
        # Conditionally add MA lines, but chart structure is maintained
        if show_20ma:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MA20'],
                mode='lines',
                name='20-Day MA',
                line=dict(color='red', width=1)
            ))
        
        if show_50ma:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MA50'],
                mode='lines',
                name='50-Day MA',
                line=dict(color='blue', width=1)
            ))
        
        if show_100ma:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MA100'],
                mode='lines',
                name='100-Day MA',
                line=dict(color='green', width=1)
            ))
        
        if show_200ma:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MA200'],
                mode='lines',
                name='200-Day MA',
                line=dict(color='purple', width=1)
            ))
        
        # Set layout
        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode="x unified"
        )
        
        # ALWAYS display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    # Display instructions
    st.info("Try toggling the Moving Average checkboxes above. The chart should always remain visible regardless of which options are selected.")
else:
    st.error("Could not load stock data. Please try again later.")