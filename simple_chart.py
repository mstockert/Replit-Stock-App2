import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Simple Stock Chart Display")
st.write("This is a minimal app to test chart display issues")

# Input for stock symbol
ticker = st.text_input("Enter Stock Symbol", "AAPL")

# Button to fetch data
if st.button("Get Stock Data"):
    # Fetch data
    data = yf.download(ticker, period="1y")
    
    if not data.empty:
        # Create base chart that should ALWAYS appear
        st.header("Basic Price Chart (Should Always Display)")
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ))
        
        fig1.update_layout(
            title=f"{ticker} Stock Price (1 Year)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
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
        
        # The key fix: ALWAYS create a moving averages chart
        # But only show selected moving averages
        
        # Display title of the moving averages section
        st.write("### Moving Average Chart")
        
        # Create figure for moving averages
        fig2 = go.Figure()
        
        # Always add price trace - this is critical
        fig2.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1.5)
        ))
        
        # Conditionally add moving averages - but the chart structure remains intact
        if show_ma20:
            fig2.add_trace(go.Scatter(
                x=data.index, 
                y=data['MA20'],
                mode='lines',
                name='20-Day MA',
                line=dict(color='red', width=1.5)
            ))
        
        if show_ma50:
            fig2.add_trace(go.Scatter(
                x=data.index, 
                y=data['MA50'],
                mode='lines',
                name='50-Day MA',
                line=dict(color='blue', width=1.5)
            ))
        
        if show_ma100:
            fig2.add_trace(go.Scatter(
                x=data.index, 
                y=data['MA100'],
                mode='lines',
                name='100-Day MA',
                line=dict(color='green', width=1.5)
            ))
        
        if show_ma200:
            fig2.add_trace(go.Scatter(
                x=data.index, 
                y=data['MA200'],
                mode='lines',
                name='200-Day MA',
                line=dict(color='purple', width=1.5)
            ))
        
        # Layout is ALWAYS applied
        fig2.update_layout(
            title=f"{ticker} with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        # ALWAYS display the chart, even if no MAs are selected
        st.plotly_chart(fig2, use_container_width=True)
        
        # Add explanation text
        st.info("Try toggling the Moving Average checkboxes above - the chart should always remain visible.")
        
        # Also display raw data for verification
        st.header("Stock Data")
        st.dataframe(data)
    else:
        st.error(f"Could not retrieve data for {ticker}")