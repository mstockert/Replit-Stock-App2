import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Simple Stock Chart Display")
st.write("This is a minimal app to test chart display issues")

# Input for stock symbol
ticker = st.text_input("Enter Stock Symbol", "AAPL")

# Button to fetch data
if st.button("Get Stock Data"):
    # Show loading message
    with st.spinner("Fetching data..."):
        # Fetch data
        data = yf.download(ticker, period="1y")
    
    if not data.empty:
        # ALWAYS show the basic price chart using Streamlit native chart
        st.header("Basic Price Chart (Should Always Display)")
        
        # The proper way to create a dataframe from a Series
        # This preserves the index from the original data
        price_df = pd.DataFrame(data['Close']).rename(columns={'Close': 'Price'})
        
        # Use Streamlit's built-in line chart (more stable than Plotly)
        st.line_chart(price_df)
        
        # Create a separate section to test checkboxes
        st.header("Moving Averages Test")
        
        # Simple checkboxes for MA periods
        col1, col2 = st.columns(2)
        with col1:
            show_ma20 = st.checkbox("20-Day MA", value=True)
            show_ma50 = st.checkbox("50-Day MA", value=True)
        with col2:
            show_ma100 = st.checkbox("100-Day MA", value=False)
            show_ma200 = st.checkbox("200-Day MA", value=True)
        
        # Calculate the moving averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA100'] = data['Close'].rolling(window=100).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # COMPLETELY DIFFERENT APPROACH:
        # Create a properly formatted display dataframe that always includes the price
        display_df = pd.DataFrame(index=data.index)
        display_df['Price'] = data['Close']  # Always add price first
        
        # Add selected MAs to the display dataframe
        if show_ma20:
            display_df['20-Day MA'] = data['MA20']
        if show_ma50:
            display_df['50-Day MA'] = data['MA50']
        if show_ma100:
            display_df['100-Day MA'] = data['MA100']
        if show_ma200:
            display_df['200-Day MA'] = data['MA200']
        
        # Display title
        st.subheader("Moving Averages Chart (Streamlit Native)")
        
        # Use Streamlit's built-in line chart (more stable than Plotly)
        st.line_chart(display_df)
        
        # Add explanation
        st.info("Try toggling the Moving Average checkboxes above - the chart should always remain visible.")
        
        # ALTERNATIVE APPROACH (using Plotly as a backup)
        st.subheader("Moving Averages Chart (Plotly)")
        
        # Create a basic figure that always includes price
        fig = go.Figure()
        
        # Always add price line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1.5)
        ))
        
        # Add MAs conditionally
        ma_colors = {
            'MA20': 'red',
            'MA50': 'blue',
            'MA100': 'green',
            'MA200': 'purple'
        }
        
        ma_display = {
            'MA20': show_ma20,
            'MA50': show_ma50,
            'MA100': show_ma100,
            'MA200': show_ma200
        }
        
        for ma_name, show in ma_display.items():
            if show:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[ma_name],
                    mode='lines',
                    name=ma_name,
                    line=dict(color=ma_colors[ma_name], width=1.5)
                ))
        
        # Apply layout
        fig.update_layout(
            title=f"{ticker} with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Also display raw data for verification
        with st.expander("View Raw Data"):
            st.dataframe(data)
    else:
        st.error(f"Could not retrieve data for {ticker}")