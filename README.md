Nifty 50 Stock Volatility & Analysis Dashboard

A comprehensive Stock Analysis Dashboard built using Python that analyzes Nifty 50 stocks using historical data from Yahoo Finance.
The project provides Technical Analysis, Fundamental Analysis, Volatility Insights, and Candlestick Pattern Detection in an interactive UI.

Features
Price & Volatility Analysis
Historical stock price visualization
Volatility calculation
Trend detection
Moving averages comparison
Bollinger Bands
Technical Analysis
RSI Indicator
MACD Indicator
SMA 20 & SMA 50
Bollinger Bands (High & Low)
Automated Trading Signals
Overall Buy / Sell Recommendation
Fundamental Analysis
Market Cap
P/E Ratio
P/B Ratio
Revenue
Profit Margin
ROE
Dividend Yield
Beta
Business Summary
Candlestick Image Analysis

Upload candlestick chart images to detect:

Doji
Hammer
Shooting Star
Engulfing patterns
Trend estimation
AI-based trading suggestions
Performance Dashboard
Interactive charts
Multi-tab navigation
Clean UI
Dark theme visualization
Tech Stack
Python
Pandas
NumPy
Matplotlib / Plotly
yfinance
Streamlit / Flask (based on your implementation)
Scikit-learn (optional for signals)
TA indicators (custom implementation)
Project Structure
stock-volatility-dashboard/
│
├── data/
├── images/
├── indicators/
├── utils/
├── app.py
├── requirements.txt
└── README.md
Data Source

This project uses:

Yahoo Finance API (via yfinance)
Nifty 50 stock symbols
Historical OHLC data
Installation

Clone the repository:

git clone https://github.com/yourusername/nifty50-stock-analysis.git
cd nifty50-stock-analysis

Install dependencies:

pip install -r requirements.txt
Run the Application
python app.py

or (if using Streamlit)

streamlit run app.py
Indicators Used
RSI
Overbought: Above 70
Oversold: Below 30
MACD
Bullish crossover
Bearish crossover
Moving Average
SMA 20
SMA 50
Trend detection
Bollinger Bands
Volatility detection
Breakout signals
Trading Signal Logic

The system generates signals based on:

RSI condition
MACD crossover
Moving average trend
Volatility levels

Final signal:

Strong Buy
Buy
Neutral
Sell
Strong Sell
Candlestick Pattern Detection

Upload chart images to detect:

Doji
Hammer
Shooting Star
Bullish Engulfing
Bearish Engulfing

AI estimates:

Market trend
Buy/Sell suggestion
Example Use Case
Analyze Nifty 50 stock volatility
Detect trend changes
Evaluate fundamentals
Identify entry/exit signals
Upload chart screenshots for pattern detection
Future Improvements
Live market data streaming
Portfolio tracker
ML-based price prediction
News sentiment analysis
Backtesting engine
Multi-timeframe analysis
