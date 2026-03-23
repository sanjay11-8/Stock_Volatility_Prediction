import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
        
# Nifty 50 stocks list
NIFTY_50_STOCKS = {
    'ADANIPORTS.NS': 'Adani Ports',
    'ASIANPAINT.NS': 'Asian Paints',
    'AXISBANK.NS': 'Axis Bank',
    'BAJAJ-AUTO.NS': 'Bajaj Auto',
    'BAJAJFINSV.NS': 'Bajaj Finserv',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'BPCL.NS': 'BPCL',
    'BRITANNIA.NS': 'Britannia',
    'CIPLA.NS': 'Cipla',
    'COALINDIA.NS': 'Coal India',
    'DIVISLAB.NS': 'Divi\'s Labs',
    'DRREDDY.NS': 'Dr Reddy\'s',
    'EICHERMOT.NS': 'Eicher Motors',
    'GRASIM.NS': 'Grasim',
    'HCLTECH.NS': 'HCL Tech',
    'HDFCBANK.NS': 'HDFC Bank',
    'HDFCLIFE.NS': 'HDFC Life',
    'HEROMOTOCO.NS': 'Hero MotoCorp',
    'HINDALCO.NS': 'Hindalco',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'ICICIBANK.NS': 'ICICI Bank',
    'ITC.NS': 'ITC',
    'INDUSINDBK.NS': 'IndusInd Bank',
    'INFY.NS': 'Infosys',
    'JSWSTEEL.NS': 'JSW Steel',
    'KOTAKBANK.NS': 'Kotak Bank',
    'LT.NS': 'L&T',
    'M&M.NS': 'M&M',
    'MARUTI.NS': 'Maruti Suzuki',
    'NESTLEIND.NS': 'Nestle India',
    'NTPC.NS': 'NTPC',
    'ONGC.NS': 'ONGC',
    'POWERGRID.NS': 'Power Grid',
    'RELIANCE.NS': 'Reliance',
    'SBILIFE.NS': 'SBI Life',
    'SBIN.NS': 'SBI',
    'SUNPHARMA.NS': 'Sun Pharma',
    'TCS.NS': 'TCS',
    'TATACONSUM.NS': 'Tata Consumer',
    'TATAMOTORS.NS': 'Tata Motors',
    'TATASTEEL.NS': 'Tata Steel',
    'TECHM.NS': 'Tech Mahindra',
    'TITAN.NS': 'Titan',
    'ULTRACEMCO.NS': 'UltraTech Cement',
    'UPL.NS': 'UPL',
    'WIPRO.NS': 'Wipro',
    'APOLLOHOSP.NS': 'Apollo Hospitals',
    'BAJAJHLDNG.NS': 'Bajaj Holdings',
    'ADANITRANS.NS': 'Adani Transmission'
}

@st.cache_data
def load_volatility_model():
    """Load the pre-trained volatility classification model"""
    try:
        with open('models/stock_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Volatility model not found. Please ensure the model file exists in the 'models' folder.")
        return None

@st.cache_data
def fetch_stock_data(symbol, period='1y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_technical_indicators(data):
    """Calculate technical indicators for the stock data"""
    # RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_High'] = bollinger.bollinger_hband()
    data['BB_Low'] = bollinger.bollinger_lband()
    data['BB_Mid'] = bollinger.bollinger_mavg()
    
    # Moving Averages
    data['SMA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
    data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
    data['EMA_12'] = ta.trend.EMAIndicator(data['Close'], window=12).ema_indicator()
    data['EMA_26'] = ta.trend.EMAIndicator(data['Close'], window=26).ema_indicator()
    
    # Volume SMA (manual calculation)
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    
    return data


def calculate_performance_metrics(data):
    """Calculate performance metrics"""
    current_price = data['Close'].iloc[-1]
    
    # Daily change
    daily_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
    
    # Weekly change (7 days)
    if len(data) >= 7:
        weekly_change = ((current_price - data['Close'].iloc[-7]) / data['Close'].iloc[-7]) * 100
    else:
        weekly_change = daily_change
    
    # Monthly change (30 days)
    if len(data) >= 30:
        monthly_change = ((current_price - data['Close'].iloc[-30]) / data['Close'].iloc[-30]) * 100
    else:
        monthly_change = weekly_change
    
    # YTD change
    ytd_start = data[data.index.year == datetime.now().year].iloc[0]['Close'] if len(data[data.index.year == datetime.now().year]) > 0 else data['Close'].iloc[0]
    ytd_change = ((current_price - ytd_start) / ytd_start) * 100
    
    # 1 year change
    if len(data) >= 252:  # Approximately 1 year of trading days
        yearly_change = ((current_price - data['Close'].iloc[-252]) / data['Close'].iloc[-252]) * 100
    else:
        yearly_change = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
    
    # 3 year change (if available)
    if len(data) >= 756:  # Approximately 3 years
        three_year_change = ((current_price - data['Close'].iloc[-756]) / data['Close'].iloc[-756]) * 100
    else:
        three_year_change = yearly_change
    
    return {
        'Daily': daily_change,
        '1 Week': weekly_change,
        '1 Month': monthly_change,
        'YTD': ytd_change,
        '1 Year': yearly_change,
        '3 Years': three_year_change
    }

def get_technical_signals(data):
    """Generate technical analysis signals"""
    latest_data = data.iloc[-1]
    
    signals = {}
    
    # RSI Signal
    try:
        rsi = latest_data['RSI']
        if pd.isna(rsi):
            signals['RSI'] = 'Neutral'
        elif rsi > 70:
            signals['RSI'] = 'Strong Sell'
        elif rsi > 60:
            signals['RSI'] = 'Sell'
        elif rsi < 30:
            signals['RSI'] = 'Strong Buy'
        elif rsi < 40:
            signals['RSI'] = 'Buy'
        else:
            signals['RSI'] = 'Neutral'
    except:
        signals['RSI'] = 'Neutral'
    
    # MACD Signal
    try:
        macd = latest_data['MACD']
        macd_signal = latest_data['MACD_Signal']
        if pd.isna(macd) or pd.isna(macd_signal):
            signals['MACD'] = 'Neutral'
        elif macd > macd_signal:
            signals['MACD'] = 'Buy' if macd > 0 else 'Strong Buy'
        else:
            signals['MACD'] = 'Sell' if macd < 0 else 'Strong Sell'
    except:
        signals['MACD'] = 'Neutral'
    
    # Moving Average Signal
    try:
        close = latest_data['Close']
        sma_20 = latest_data['SMA_20']
        sma_50 = latest_data['SMA_50']
        
        if pd.isna(sma_20) or pd.isna(sma_50):
            signals['Moving Average'] = 'Neutral'
        elif close > sma_20 > sma_50:
            signals['Moving Average'] = 'Strong Buy'
        elif close > sma_20:
            signals['Moving Average'] = 'Buy'
        elif close < sma_20 < sma_50:
            signals['Moving Average'] = 'Strong Sell'
        elif close < sma_20:
            signals['Moving Average'] = 'Sell'
        else:
            signals['Moving Average'] = 'Neutral'
    except:
        signals['Moving Average'] = 'Neutral'
    
    # Overall Signal (simplified logic)
    buy_signals = sum(1 for signal in signals.values() if 'Buy' in signal)
    sell_signals = sum(1 for signal in signals.values() if 'Sell' in signal)
    
    if buy_signals > sell_signals:
        overall_signal = 'Strong Buy' if buy_signals >= 2 else 'Buy'
    elif sell_signals > buy_signals:
        overall_signal = 'Strong Sell' if sell_signals >= 2 else 'Sell'
    else:
        overall_signal = 'Neutral'
    
    return signals, overall_signal

def predict_volatility(data, model):
    """Predict volatility using the pre-trained pipeline"""
    if model is None:
        return "Model not available", 0
    
    try:
        # Feature engineering (MUST match training!)
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
        data['Volume_Change'] = data['Volume'].pct_change()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        
        latest_features = data[['Volatility','High_Low_Pct','Volume_Change','RSI','SMA_20','SMA_50','MACD']].iloc[-1:].fillna(0)
        
        # Directly predict with pipeline (no manual scaling)
        prediction = model.predict(latest_features)[0]
        prediction_proba = model.predict_proba(latest_features)[0]
        
        volatility_classes = ['Low Volatility', 'Medium Volatility', 'High Volatility']
        return volatility_classes[prediction], max(prediction_proba)
    
    except Exception as e:
        return f"Prediction error: {e}", 0



def create_price_chart(data, symbol):
    """Create an interactive price chart"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol
    ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA_20'],
        mode='lines',
        name='SMA 20',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA_50'],
        mode='lines',
        name='SMA 50',
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title=f'{symbol} Stock Price',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_white',
        height=500
    )
    
    return fig

def main():
    st.set_page_config(page_title="Dashboard", layout="wide")
    
    st.title("Dashboard")
    st.markdown("---")
    
    # Load volatility model
    volatility_model = load_volatility_model()
    
    # Sidebar for stock selection
    st.sidebar.header("📊 Stock Selection")
    selected_stock = st.sidebar.selectbox(
        "Choose a Nifty 50 Stock:",
        list(NIFTY_50_STOCKS.keys()),
        format_func=lambda x: f"{NIFTY_50_STOCKS[x]} ({x.replace('.NS', '')})"
    )
    
    # Time period selection
    period_options = {
        '1D': '1d',
        '1W': '5d',
        '1M': '1mo',
        '6M': '6mo',
        '1Y': '1y',
        'Max': 'max'
    }
    
    selected_period = st.sidebar.selectbox(
        "Select Time Period:",
        list(period_options.keys()),
        index=4
    )
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
    
    # Fetch and process data
    with st.spinner(f"Loading data for {NIFTY_50_STOCKS[selected_stock]}..."):
        data = fetch_stock_data(selected_stock, period_options[selected_period])
        
        if data is not None and not data.empty:
            # Calculate technical indicators
            data = calculate_technical_indicators(data)
            
            # Main dashboard layout
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            with col1:
                st.metric(
                    label="💰 Current Price",
                    value=f"₹{current_price:.2f}",
                    delta=f"{price_change_pct:.2f}%"
                )
            
            with col2:
                high_52w = data['High'].rolling(window=252).max().iloc[-1]
                st.metric(
                    label="📈 52W High",
                    value=f"₹{high_52w:.2f}"
                )
            
            with col3:
                low_52w = data['Low'].rolling(window=252).min().iloc[-1]
                st.metric(
                    label="📉 52W Low",
                    value=f"₹{low_52w:.2f}"
                )
            
            with col4:
                volume = data['Volume'].iloc[-1]
                st.metric(
                    label="📊 Volume",
                    value=f"{volume:,.0f}"
                )
            
            # Volatility Prediction
            # Volatility Prediction
            # Volatility Prediction
            if volatility_model is not None:
                vol_prediction, confidence = predict_volatility(data, volatility_model)

                # Color mapping
                color_map = {
                    "Low Volatility": "green",
                    "Medium Volatility": "orange",
                    "High Volatility": "red"
                }

                color = color_map.get(vol_prediction, "black")

                st.markdown(
                    f"""
                    <div style="padding:10px; border-radius:8px; background-color:{color}; color:white; font-weight:bold; text-align:center;">
                        🎯 Predicted Volatility: {vol_prediction} <br>
                        Confidence: {confidence:.2%}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Performance", "🔧 Technical Analysis", "📋 Fundamental"])
            
            with tab1:
                # Price chart
                chart = create_price_chart(data, selected_stock)
                st.plotly_chart(chart, use_container_width=True)
                
                # Recent price data
                st.subheader("Recent Price Data")
                recent_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
                st.dataframe(recent_data)
            
            with tab2:
                st.subheader("Performance Metrics")
                
                # Calculate performance
                performance = calculate_performance_metrics(data)
                
                # Create performance dataframe
                perf_df = pd.DataFrame({
                    'Period': list(performance.keys()),
                    'Return (%)': [f"{v:.2f}%" for v in performance.values()],
                    'Return Value': list(performance.values())
                })
                
                # Color code the performance
                def color_performance(val):
                    if val > 0:
                        return 'color: green'
                    elif val < 0:
                        return 'color: red'
                    else:
                        return 'color: black'
                
                styled_df = perf_df.style.applymap(color_performance, subset=['Return Value'])
                st.dataframe(styled_df, hide_index=True)
                
                # Performance chart
                fig_perf = px.bar(
                    perf_df, 
                    x='Period', 
                    y='Return Value',
                    color='Return Value',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    title="Performance Across Different Time Periods"
                )
                st.plotly_chart(fig_perf, use_container_width=True)
            
            with tab3:
                st.subheader("Technical Analysis")
                
                # Technical indicators
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Current Technical Indicators:**")
                    latest = data.iloc[-1]
                    
                    indicators_df = pd.DataFrame({
                        'Indicator': ['RSI', 'MACD', 'MACD Signal', 'SMA 20', 'SMA 50', 'BB High', 'BB Low'],
                        'Value': [
                            f"{latest['RSI']:.2f}",
                            f"{latest['MACD']:.2f}",
                            f"{latest['MACD_Signal']:.2f}",
                            f"₹{latest['SMA_20']:.2f}",
                            f"₹{latest['SMA_50']:.2f}",
                            f"₹{latest['BB_High']:.2f}",
                            f"₹{latest['BB_Low']:.2f}"
                        ]
                    })
                    
                    st.dataframe(indicators_df, hide_index=True)
                
                with col2:
                    st.write("**Technical Signals:**")
                    signals, overall = get_technical_signals(data)
                    
                    signals_df = pd.DataFrame({
                        'Timeframe': ['Hourly', 'Daily', 'Weekly', 'Monthly'],
                        'Signal': [overall, overall, signals.get('Moving Average', 'Neutral'), signals.get('RSI', 'Neutral')]
                    })
                    
                    def color_signals(val):
                        if 'Buy' in val:
                            return 'color: green; font-weight: bold'
                        elif 'Sell' in val:
                            return 'color: red; font-weight: bold'
                        else:
                            return 'color: orange; font-weight: bold'
                    
                    styled_signals = signals_df.style.applymap(color_signals, subset=['Signal'])
                    st.dataframe(styled_signals, hide_index=True)
                
                # RSI Chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(title="RSI Indicator", yaxis_title="RSI", height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with tab4:
                st.subheader("Fundamental Analysis")
                
                # Fetch additional fundamental data
                try:
                    stock_info = yf.Ticker(selected_stock).info
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Valuation Metrics:**")
                        if 'marketCap' in stock_info:
                            st.write(f"Market Cap: ₹{stock_info['marketCap']:,.0f}")
                        if 'trailingPE' in stock_info:
                            st.write(f"P/E Ratio: {stock_info['trailingPE']:.2f}")
                        if 'priceToBook' in stock_info:
                            st.write(f"P/B Ratio: {stock_info['priceToBook']:.2f}")
                    
                    with col2:
                        st.write("**Financial Metrics:**")
                        if 'totalRevenue' in stock_info:
                            st.write(f"Revenue: ₹{stock_info['totalRevenue']:,.0f}")
                        if 'profitMargins' in stock_info:
                            st.write(f"Profit Margin: {stock_info['profitMargins']:.2%}")
                        if 'returnOnEquity' in stock_info:
                            st.write(f"ROE: {stock_info['returnOnEquity']:.2%}")
                    
                    with col3:
                        st.write("**Dividend & Yield:**")
                        if 'dividendYield' in stock_info and stock_info['dividendYield']:
                            st.write(f"Dividend Yield: {stock_info['dividendYield']:.2%}")
                        if 'payoutRatio' in stock_info:
                            st.write(f"Payout Ratio: {stock_info['payoutRatio']:.2%}")
                    
                    # Company description
                    if 'longBusinessSummary' in stock_info:
                        st.write("**Business Summary:**")
                        st.write(stock_info['longBusinessSummary'][:500] + "...")
                
                except Exception as e:
                    st.error(f"Error fetching fundamental data: {e}")
        
        else:
            st.error("Failed to fetch stock data. Please try again.")

if __name__ == "__main__":
    main()