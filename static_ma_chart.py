import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Chart with Moving Averages", layout="wide")

st.title("Stock Chart with Moving Averages")
st.write("A reliable chart display that always shows the data")

# Input for ticker symbol
ticker = st.text_input("Enter Stock Symbol", "AAPL")

# Period selector
period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo", 
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "Max": "max"
}
selected_period_label = st.selectbox("Select Time Period", 
                                    options=list(period_options.keys()),
                                    index=3)  # Default to 1 Year
selected_period = period_options[selected_period_label]

# Button to fetch data
if st.button("Get Stock Data"):
    # Show loading indicator
    with st.spinner(f"Fetching {ticker} data..."):
        try:
            # Fetch data
            data = yf.download(ticker, period=selected_period)
            
            if data.empty:
                st.error(f"No data found for {ticker}")
            else:
                # Display company info
                st.subheader(f"{ticker} Stock Data")
                
                # Basic data display
                st.write(f"**Time Period:** {selected_period_label}")
                st.write(f"**Data Points:** {len(data)}")
                st.write(f"**Price Range:** ${data['Low'].min():.2f} - ${data['High'].max():.2f}")
                
                # Calculate all MAs upfront
                data['MA20'] = data['Close'].rolling(window=20).mean()
                data['MA50'] = data['Close'].rolling(window=50).mean()
                data['MA100'] = data['Close'].rolling(window=100).mean()
                data['MA200'] = data['Close'].rolling(window=200).mean()
                
                # Create tabs
                tab1, tab2 = st.tabs(["Moving Averages", "Price Data"])
                
                with tab1:
                    # Use radio buttons instead of checkboxes
                    st.write("### Select Moving Averages to Display")
                    ma_selection = st.multiselect(
                        "Choose Moving Averages", 
                        ["20-Day MA", "50-Day MA", "100-Day MA", "200-Day MA"],
                        default=["20-Day MA", "50-Day MA"]
                    )
                    
                    # Always create the basic chart
                    fig = go.Figure()
                    
                    # Always add price data - this never changes
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='black', width=2)
                    ))
                    
                    # Add selected MAs
                    ma_mapping = {
                        "20-Day MA": {"column": "MA20", "color": "red"},
                        "50-Day MA": {"column": "MA50", "color": "blue"},
                        "100-Day MA": {"column": "MA100", "color": "green"},
                        "200-Day MA": {"column": "MA200", "color": "purple"}
                    }
                    
                    for ma in ma_selection:
                        ma_info = ma_mapping[ma]
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data[ma_info["column"]],
                            mode='lines',
                            name=ma,
                            line=dict(color=ma_info["color"], width=1.5)
                        ))
                    
                    # Set layout
                    fig.update_layout(
                        title=f"{ticker} Stock Price with Moving Averages",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=600,
                        hovermode="x unified"
                    )
                    
                    # ALWAYS display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Also show a simpler Streamlit native chart as backup
                    st.subheader("Alternative Chart (Streamlit Native)")
                    
                    # Create display dataframe
                    display_df = pd.DataFrame(index=data.index)
                    display_df['Price'] = data['Close']
                    
                    # Add selected MAs
                    for ma in ma_selection:
                        column = ma_mapping[ma]["column"]
                        display_df[ma] = data[column]
                    
                    # Display chart
                    st.line_chart(display_df)
                
                with tab2:
                    # Show raw data
                    st.dataframe(data)
                    
                    # Download button
                    csv = data.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{ticker}_{selected_period}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error retrieving data: {str(e)}")
            st.exception(e)