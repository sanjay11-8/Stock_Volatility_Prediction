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
import cv2
from PIL import Image
warnings.filterwarnings('ignore')

st.set_page_config(
        page_title="Stock Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📈"
    )

# Enhanced CSS with better contrast and attractive styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2563eb;
        --secondary-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
        --bg-color: #0f172a;
        --card-bg: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        background-attachment: fixed;
    }
    
    /* Content area */
    .main .block-container {
        background-color: rgba(30, 41, 59, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid #334155;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 2px solid #334155;
    }
    
    /* All text elements */
    .stApp, .main, .block-container, [data-testid="stSidebar"] {
        color: var(--text-primary) !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 700;
    }
    
    h1 {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Paragraphs and labels */
    p, label, div, span {
        color: var(--text-primary) !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        color: var(--text-secondary) !important;
        font-weight: 600;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #475569;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        border-color: #60a5fa;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #334155;
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: var(--text-secondary) !important;
        font-weight: 600;
        padding: 12px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(96, 165, 250, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(96, 165, 250, 0.4);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #1e293b !important;
        border: 2px solid #475569 !important;
        border-radius: 10px;
        color: var(--text-primary) !important;
    }
    
    .stSelectbox label {
        color: var(--text-secondary) !important;
    }
    
    /* DataFrames */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #1e293b !important;
    }
    
    .dataframe th {
        background-color: #334155 !important;
        color: var(--text-primary) !important;
    }
    
    .dataframe td {
        background-color: #1e293b !important;
        color: var(--text-primary) !important;
        border-color: #475569 !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1e293b !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 12px;
        padding: 1rem;
        color: var(--text-primary) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1e293b !important;
        border: 2px dashed #475569 !important;
        border-radius: 12px;
        padding: 2rem;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: var(--text-primary) !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #475569, transparent);
        margin: 2rem 0;
    }
    
    /* Custom card class */
    .custom-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid #475569;
        margin-bottom: 1rem;
        color: var(--text-primary) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #60a5fa !important;
    }
    
    /* Text input */
    .stTextInput input {
        background-color: #1e293b !important;
        border: 2px solid #475569 !important;
        color: var(--text-primary) !important;
        border-radius: 10px;
    }
    
    /* Number input */
    .stNumberInput input {
        background-color: #1e293b !important;
        border: 2px solid #475569 !important;
        color: var(--text-primary) !important;
        border-radius: 10px;
    }
    
    /* Date input */
    .stDateInput input {
        background-color: #1e293b !important;
        border: 2px solid #475569 !important;
        color: var(--text-primary) !important;
        border-radius: 10px;
    }
    
    /* Multi-select */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #334155 !important;
        color: var(--text-primary) !important;
    }
    
    .stMultiSelect [data-baseweb="select"] {
        background-color: #1e293b !important;
        border: 2px solid #475569 !important;
        color: var(--text-primary) !important;
    }
    
    /* Radio buttons */
    .stRadio [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }
    
    /* Checkbox */
    .stCheckbox [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }
    
    /* Success, Info, Warning, Error messages */
    .stSuccess {
        background-color: #064e3b !important;
        border: 2px solid #10b981 !important;
        color: var(--text-primary) !important;
    }
    
    .stInfo {
        background-color: #1e3a8a !important;
        border: 2px solid #3b82f6 !important;
        color: var(--text-primary) !important;
    }
    
    .stWarning {
        background-color: #78350f !important;
        border: 2px solid #f59e0b !important;
        color: var(--text-primary) !important;
    }
    
    .stError {
        background-color: #7f1d1d !important;
        border: 2px solid #ef4444 !important;
        color: var(--text-primary) !important;
    }
    
    /* Plotly chart background override */
    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }
    
    /* Sidebar specific styling */
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: var(--text-secondary) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #60a5fa;
    }
</style>
""", unsafe_allow_html=True)

# Nifty 50 stocks list (unchanged)
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

# ========== IMAGE ANALYSIS FUNCTIONS ==========
# (All image analysis functions remain exactly the same)
def to_np(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def resize_keep_aspect(img, max_width=1200):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def detect_candles(img_bgr):
    """Returns list of detected candles with position and characteristics"""
    img = img_bgr.copy()
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 10)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    opened = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_v, iterations=1)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 50 or cw > 0.2 * w or ch < 10:
            continue
        candidate_boxes.append((x, y, cw, ch))

    candidate_boxes = sorted(candidate_boxes, key=lambda b: b[0])

    candles = []
    for (x, y, cw, ch) in candidate_boxes:
        pad_x = max(2, cw // 8)
        pad_y = max(2, ch // 8)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + cw + pad_x)
        y1 = min(h, y + ch + pad_y)
        box = img[y0:y1, x0:x1]

        hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
        mask_g = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
        mask_r1 = cv2.inRange(hsv, (0, 40, 40), (10, 255, 255))
        mask_r2 = cv2.inRange(hsv, (160, 40, 40), (179, 255, 255))
        mask_r = cv2.bitwise_or(mask_r1, mask_r2)

        mask_color = cv2.bitwise_or(mask_g, mask_r)
        if cv2.countNonZero(mask_color) < 3:
            v = hsv[:, :, 2]
            _, mask_bright = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)
            _, mask_dark = cv2.threshold(v, 50, 255, cv2.THRESH_BINARY_INV)
            mask_g = mask_bright
            mask_r = mask_dark

        combined = cv2.bitwise_or(mask_g, mask_r)
        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        body_top, body_bottom = None, None
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c)
            body_top = y0 + by
            body_bottom = y0 + by + bh
            body_mask = np.zeros_like(combined)
            cv2.drawContours(body_mask, [c], -1, 255, -1)
            g_count = int(cv2.countNonZero(cv2.bitwise_and(body_mask, mask_g)))
            r_count = int(cv2.countNonZero(cv2.bitwise_and(body_mask, mask_r)))
            color = 'bull' if g_count >= r_count else 'bear'
        else:
            body_top = y0 + int((y1 - y0) * 0.35)
            body_bottom = y0 + int((y1 - y0) * 0.65)
            color = 'bull'

        cx = x0 + (x1 - x0) // 2
        column = gray[:, cx]
        non_bg = np.where(column < 245)[0]
        if len(non_bg) == 0:
            high, low = y, y + ch
        else:
            high, low = int(non_bg.min()), int(non_bg.max())

        candle = {
            'x': (x0 + x1) // 2,
            'x0': x0, 'x1': x1,
            'w': (x1 - x0),
            'body_top': int(body_top),
            'body_bottom': int(body_bottom),
            'high': int(high),
            'low': int(low),
            'color': color
        }
        candles.append(candle)

    filtered = []
    last_x = -9999
    for c in candles:
        if c['x'] - last_x < max(4, c['w']//2):
            continue
        filtered.append(c)
        last_x = c['x']
    return filtered

def candle_body_height(c):
    return abs(c['body_bottom'] - c['body_top'])

def candle_total_range(c):
    return abs(c['low'] - c['high'])

def detect_doji(c, thresh_ratio=0.2):
    total = candle_total_range(c)
    body = candle_body_height(c)
    if total == 0:
        return False
    return body <= thresh_ratio * total

def detect_hammer(c):
    total = candle_total_range(c)
    body = candle_body_height(c)
    if total == 0:
        return False
    lower_wick = min(c['body_top'], c['body_bottom']) - c['low']
    upper_wick = c['high'] - max(c['body_top'], c['body_bottom'])
    return (lower_wick >= 2 * body) and (body <= 0.35 * total)

def detect_shooting_star(c):
    total = candle_total_range(c)
    body = candle_body_height(c)
    if total == 0:
        return False
    upper_wick = c['high'] - max(c['body_top'], c['body_bottom'])
    return (upper_wick >= 2 * body) and (body <= 0.35 * total)

def detect_engulfing(prev_c, cur_c):
    prev_top = min(prev_c['body_top'], prev_c['body_bottom'])
    prev_bottom = max(prev_c['body_top'], prev_c['body_bottom'])
    cur_top = min(cur_c['body_top'], cur_c['body_bottom'])
    cur_bottom = max(cur_c['body_top'], cur_c['body_bottom'])
    if prev_c['color'] == cur_c['color']:
        return None
    if (cur_top <= prev_top) and (cur_bottom >= prev_bottom) and (abs(cur_bottom - cur_top) > abs(prev_bottom - prev_top)):
        return 'bullish' if cur_c['color'] == 'bull' else 'bearish'
    return None

def estimate_close(c):
    if c['color'] == 'bull':
        return min(c['body_top'], c['body_bottom'])
    else:
        return max(c['body_top'], c['body_bottom'])

def trend_from_closes(closes):
    if len(closes) < 3:
        return 'sideways', 0.0
    xs = np.arange(len(closes))
    coeffs = np.polyfit(xs, closes, 1)
    slope = coeffs[0]
    if slope < -0.5:
        return 'uptrend', slope
    elif slope > 0.5:
        return 'downtrend', slope
    else:
        return 'sideways', slope

def analyze(candles):
    results = []
    for i, c in enumerate(candles):
        r = {'index': i, 'kind': None, 'desc': None}
        if detect_doji(c):
            r['kind'] = 'Doji'
            r['desc'] = 'Small body than wicks: market indecision.'
        elif detect_hammer(c):
            r['kind'] = 'Hammer'
            r['desc'] = 'Long lower wick and small body: potential bullish reversal after downtrend.'
        elif detect_shooting_star(c):
            r['kind'] = 'Shooting Star'
            r['desc'] = 'Long upper wick and small body: potential bearish reversal after uptrend.'
        else:
            r['kind'] = None
        results.append(r)

    engulfs = []
    for i in range(1, len(candles)):
        ev = detect_engulfing(candles[i-1], candles[i])
        if ev:
            engulfs.append({'index': i, 'type': ev, 'desc': f'{ev.capitalize()} Engulfing'})

    closes = [estimate_close(c) for c in candles]
    trend, slope = trend_from_closes(closes[-8:])
    suggestion = 'HOLD'
    reasons = []

    bullish_patterns = [r for r in results if r['kind'] in ('Hammer',)]
    bearish_patterns = [r for r in results if r['kind'] in ('Shooting Star',)]
    bullish_engulf = [e for e in engulfs if e['type'] == 'bullish']
    bearish_engulf = [e for e in engulfs if e['type'] == 'bearish']

    if bullish_engulf or bullish_patterns:
        if trend in ('downtrend', 'sideways'):
            suggestion = 'BUY'
            reasons.append('Bullish pattern(s) detected and trend not strongly up.')
        else:
            suggestion = 'HOLD'
            reasons.append('Bullish pattern(s) but trend is already up — consider waiting for confirmation.')

    if bearish_engulf or bearish_patterns:
        suggestion = 'SELL'
        reasons.append('Bearish pattern(s) detected.')

    if not reasons:
        reasons.append('No decisive pattern detected — market may be sideways or image unclear.')

    return {
        'pattern_results': results,
        'engulfing': engulfs,
        'trend': trend,
        'slope': float(slope),
        'suggestion': suggestion,
        'reasons': reasons
    }

def annotate_image(img_bgr, candles, analysis):
    img = img_bgr.copy()
    h, w = img.shape[:2]
    for i, c in enumerate(candles):
        color = (0,255,0) if c['color']=='bull' else (0,0,255)
        cv2.rectangle(img, (c['x0'], c['body_top']), (c['x1'], c['body_bottom']), color, 2)
        cv2.line(img, (c['x'], c['high']), (c['x'], c['low']), (100,100,100), 1)
        cv2.putText(img, str(i), (c['x']-8, c['body_top']-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    for r in analysis['pattern_results']:
        if r['kind']:
            c = candles[r['index']]
            cv2.putText(img, r['kind'], (c['x']-30, c['low']+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

    for e in analysis['engulfing']:
        c = candles[e['index']]
        cv2.putText(img, e['type'] + ' Engulf', (c['x']-40, c['high']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 2)

    overlay = img.copy()
    box_h = 90
    cv2.rectangle(overlay, (5,5), (min(w-5,500), 5+box_h), (0,0,0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    txt_x = 10
    txt_y = 25
    summary_lines = [
        f"Trend: {analysis['trend']} (slope={analysis['slope']:.2f})",
        f"Suggestion: {analysis['suggestion']}",
    ] + analysis['reasons'][:3]
    for i, line in enumerate(summary_lines):
        cv2.putText(img, line, (txt_x, txt_y + 18*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return img

# ========== ORIGINAL DASHBOARD FUNCTIONS ==========
# (All original dashboard functions remain exactly the same)
@st.cache_data
def load_volatility_model():
    """Load the pre-trained volatility classification model"""
    try:
        with open('models/stock_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning("⚠️ Volatility model not found. Some features will be limited.")
        return None

@st.cache_data
def fetch_stock_data(symbol, period='1y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except Exception as e:
        st.error(f"❌ Error fetching data for {symbol}: {e}")
        return None

def calculate_technical_indicators(data):
    """Calculate technical indicators for the stock data"""
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Histogram'] = macd.macd_diff()
    
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_High'] = bollinger.bollinger_hband()
    data['BB_Low'] = bollinger.bollinger_lband()
    data['BB_Mid'] = bollinger.bollinger_mavg()
    
    data['SMA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
    data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
    data['EMA_12'] = ta.trend.EMAIndicator(data['Close'], window=12).ema_indicator()
    data['EMA_26'] = ta.trend.EMAIndicator(data['Close'], window=26).ema_indicator()
    
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    
    return data

def calculate_performance_metrics(data):
    """Calculate performance metrics"""
    current_price = data['Close'].iloc[-1]
    
    daily_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
    
    if len(data) >= 7:
        weekly_change = ((current_price - data['Close'].iloc[-7]) / data['Close'].iloc[-7]) * 100
    else:
        weekly_change = daily_change
    
    if len(data) >= 30:
        monthly_change = ((current_price - data['Close'].iloc[-30]) / data['Close'].iloc[-30]) * 100
    else:
        monthly_change = weekly_change
    
    ytd_start = data[data.index.year == datetime.now().year].iloc[0]['Close'] if len(data[data.index.year == datetime.now().year]) > 0 else data['Close'].iloc[0]
    ytd_change = ((current_price - ytd_start) / ytd_start) * 100
    
    if len(data) >= 252:
        yearly_change = ((current_price - data['Close'].iloc[-252]) / data['Close'].iloc[-252]) * 100
    else:
        yearly_change = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
    
    if len(data) >= 756:
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
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
        data['Volume_Change'] = data['Volume'].pct_change()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        
        latest_features = data[['Volatility','High_Low_Pct','Volume_Change','RSI','SMA_20','SMA_50','MACD']].iloc[-1:].fillna(0)
        
        prediction = model.predict(latest_features)[0]
        prediction_proba = model.predict_proba(latest_features)[0]
        
        volatility_classes = ['Low Volatility', 'Medium Volatility', 'High Volatility']
        return volatility_classes[prediction], max(prediction_proba)
    
    except Exception as e:
        return f"Prediction error: {e}", 0

def create_price_chart(data, symbol):
    """Create an interactive price chart with dark theme"""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol,
        increasing_line_color='#10b981',
        decreasing_line_color='#ef4444'
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA_20'],
        mode='lines',
        name='SMA 20',
        line=dict(color='#f59e0b', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA_50'],
        mode='lines',
        name='SMA 50',
        line=dict(color='#3b82f6', width=2)
    ))
    
    fig.update_layout(
        title=dict(
            text=f'{symbol} Stock Price Chart',
            font=dict(size=24, color='#f1f5f9', family='Arial Black')
        ),
        yaxis_title='Price (₹)',
        xaxis_title='Date',
        template='plotly_dark',
        height=550,
        paper_bgcolor='rgba(30,41,59,0.95)',
        plot_bgcolor='rgba(15,23,42,0.5)',
        font=dict(color='#f1f5f9'),
        xaxis=dict(
            showgrid=True,
            gridcolor='#475569',
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#475569',
            gridwidth=1
        ),
        hovermode='x unified'
    )
    
    return fig

def main():
    
    # Title with custom styling
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0 2rem 0;'>
            <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'> Stock Analysis Dashboard</h1>
            <p style='color: #cbd5e1; font-size: 1.1rem;'>Advanced Technical & Fundamental Analysis for Nifty 50 Stocks</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    volatility_model = load_volatility_model()
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("###  Stock Selection")
        selected_stock = st.selectbox(
            "Choose a Nifty 50 Stock:",
            list(NIFTY_50_STOCKS.keys()),
            format_func=lambda x: f"{NIFTY_50_STOCKS[x]} ({x.replace('.NS', '')})"
        )
        
        st.markdown("###  Time Period")
        period_options = {
            '1W': '5d',
            '1M': '1mo',
            '6M': '6mo',
            '1Y': '1y',
            'Max': 'max'
        }
        
        selected_period = st.selectbox(
            "Select Time Period:",
            list(period_options.keys()),
            index=4
        )
        
        st.markdown("---")
        
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("Data refreshed!")
        
        st.markdown("---")
        st.markdown("""
            <div style='padding: 1rem; background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%); 
                        border-radius: 10px; color: white; text-align: center;'>
                <p style='margin: 0; font-size: 0.9rem;'> <b>Pro Tip</b></p>
                <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem;'>
                    Combine multiple indicators for better trading decisions
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with st.spinner(f"🔍 Loading data for {NIFTY_50_STOCKS[selected_stock]}..."):
        data = fetch_stock_data(selected_stock, period_options[selected_period])
        
        if data is not None and not data.empty:
            data = calculate_technical_indicators(data)
            
            # Key Metrics Cards
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"₹{current_price:.2f}",
                    delta=f"{price_change_pct:.2f}%"
                )
            
            with col2:
                high_52w = data['High'].rolling(window=252).max().iloc[-1]
                st.metric(
                    label="52W High",
                    value=f"₹{high_52w:.2f}"
                )
            
            with col3:
                low_52w = data['Low'].rolling(window=252).min().iloc[-1]
                st.metric(
                    label="52W Low",
                    value=f"₹{low_52w:.2f}"
                )
            
            with col4:
                volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
                volume_change = ((volume - avg_volume) / avg_volume) * 100
                st.metric(
                    label="Volume",
                    value=f"{volume/1000000:.2f}M",
                    delta=f"{volume_change:.1f}% vs avg"
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Volatility Prediction Card
            if volatility_model is not None:
                vol_prediction, confidence = predict_volatility(data, volatility_model)
                color_map = {
                    "Low Volatility": "linear-gradient(135deg, #10b981 0%, #059669 100%)",
                    "Medium Volatility": "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
                    "High Volatility": "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
                }
                gradient = color_map.get(vol_prediction, "linear-gradient(135deg, #64748b 0%, #475569 100%)")
                
                st.markdown(
                    f"""
                    <div style="padding: 1.5rem; border-radius: 15px; background: {gradient}; 
                                color: white; text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.15);">
                        <h3 style='margin: 0 0 0.5rem 0; color: white;'>AI Volatility Prediction</h3>
                        <p style='margin: 0; font-size: 1.8rem; font-weight: bold;'>{vol_prediction}</p>
                        <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;'>
                            Confidence: {confidence:.1%}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Enhanced Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Price Chart", 
                "Performance", 
                "Technical Analysis", 
                "Fundamental",
                "Image Analysis"
            ])
            
            with tab1:
                st.markdown("### Interactive Price Chart")
                chart = create_price_chart(data, selected_stock)
                st.plotly_chart(chart, use_container_width=True)
                
                st.markdown("### Recent Price Data")
                recent_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
                recent_data.index = recent_data.index.strftime('%Y-%m-%d')
                st.dataframe(recent_data.style.format({
                    'Open': '₹{:.2f}',
                    'High': '₹{:.2f}',
                    'Low': '₹{:.2f}',
                    'Close': '₹{:.2f}',
                    'Volume': '{:,.0f}'
                }), use_container_width=True)
            
            with tab2:
                st.markdown("### Performance Metrics")
                
                performance = calculate_performance_metrics(data)
                
                # Create performance cards
                perf_cols = st.columns(3)
                perf_items = list(performance.items())
                
                for idx, (period, value) in enumerate(perf_items[:3]):
                    with perf_cols[idx]:
                        color = "#10b981" if value > 0 else "#ef4444" if value < 0 else "#cbd5e1"
                        st.markdown(f"""
                            <div class='custom-card'>
                                <p style='color: #cbd5e1; margin: 0; font-size: 0.9rem;'>{period}</p>
                                <p style='color: {color}; margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: bold;'>
                                    {value:+.2f}%
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                
                perf_cols2 = st.columns(3)
                for idx, (period, value) in enumerate(perf_items[3:]):
                    with perf_cols2[idx]:
                        color = "#10b981" if value > 0 else "#ef4444" if value < 0 else "#cbd5e1"
                        st.markdown(f"""
                            <div class='custom-card'>
                                <p style='color: #cbd5e1; margin: 0; font-size: 0.9rem;'>{period}</p>
                                <p style='color: {color}; margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: bold;'>
                                    {value:+.2f}%
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Performance Chart
                perf_df = pd.DataFrame({
                    'Period': list(performance.keys()),
                    'Return (%)': list(performance.values())
                })
                
                fig_perf = px.bar(
                    perf_df, 
                    x='Period', 
                    y='Return (%)',
                    color='Return (%)',
                    color_continuous_scale=['#ef4444', '#fbbf24', '#10b981'],
                    title="Performance Comparison Across Time Periods"
                )
                fig_perf.update_layout(
                    template='plotly_dark',
                    height=450,
                    paper_bgcolor='rgba(30,41,59,0.95)',
                    plot_bgcolor='rgba(15,23,42,0.5)',
                    font=dict(color='#f1f5f9', size=12),
                    title_font=dict(size=20, color='#f1f5f9')
                )
                st.plotly_chart(fig_perf, use_container_width=True)
            
            with tab3:
                st.markdown("### 🔧 Technical Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Current Technical Indicators")
                    latest = data.iloc[-1]
                    
                    indicators_data = {
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
                    }
                    
                    indicators_df = pd.DataFrame(indicators_data)
                    st.dataframe(indicators_df, hide_index=True, use_container_width=True)
                
                with col2:
                    st.markdown("#### Trading Signals")
                    signals, overall = get_technical_signals(data)
                    
                    # Overall signal card
                    signal_color = {
                        'Strong Buy': '#10b981',
                        'Buy': '#34d399',
                        'Neutral': '#f59e0b',
                        'Sell': '#fca5a5',
                        'Strong Sell': '#ef4444'
                    }.get(overall, '#64748b')
                    
                    st.markdown(f"""
                        <div style='padding: 1.5rem; background: {signal_color}; color: white; 
                                    border-radius: 12px; text-align: center; margin-bottom: 1rem;'>
                            <h3 style='margin: 0; color: white;'>Overall Signal</h3>
                            <p style='margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: bold;'>{overall}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Individual signals
                    for indicator, signal in signals.items():
                        sig_color = {
                            'Strong Buy': '#10b981',
                            'Buy': '#34d399',
                            'Neutral': '#f59e0b',
                            'Sell': '#fca5a5',
                            'Strong Sell': '#ef4444'
                        }.get(signal, '#64748b')
                        
                        st.markdown(f"""
                            <div style='padding: 0.75rem; background: #1e293b; border-left: 4px solid {sig_color}; 
                                        border-radius: 8px; margin-bottom: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                                <span style='color: #cbd5e1; font-size: 0.9rem;'>{indicator}:</span>
                                <span style='color: {sig_color}; font-weight: bold; float: right;'>{signal}</span>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # RSI Chart
                st.markdown("#### RSI Indicator")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=data.index, 
                    y=data['RSI'], 
                    name='RSI',
                    line=dict(color='#60a5fa', width=2)
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10b981", annotation_text="Oversold (30)")
                fig_rsi.add_hrect(y0=70, y1=100, fillcolor="#ef4444", opacity=0.1)
                fig_rsi.add_hrect(y0=0, y1=30, fillcolor="#10b981", opacity=0.1)
                fig_rsi.update_layout(
                    yaxis_title="RSI Value",
                    template='plotly_dark',
                    height=350,
                    paper_bgcolor='rgba(30,41,59,0.95)',
                    plot_bgcolor='rgba(15,23,42,0.5)',
                    font=dict(color='#f1f5f9'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # MACD Chart
                st.markdown("#### MACD Indicator")
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    name='MACD',
                    line=dict(color='#3b82f6', width=2)
                ))
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    name='Signal',
                    line=dict(color='#f59e0b', width=2)
                ))
                fig_macd.add_trace(go.Bar(
                    x=data.index,
                    y=data['MACD_Histogram'],
                    name='Histogram',
                    marker_color='#94a3b8'
                ))
                fig_macd.update_layout(
                    yaxis_title="MACD Value",
                    template='plotly_dark',
                    height=350,
                    paper_bgcolor='rgba(30,41,59,0.95)',
                    plot_bgcolor='rgba(15,23,42,0.5)',
                    font=dict(color='#f1f5f9'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_macd, use_container_width=True)
            
            with tab4:
                st.markdown("### Fundamental Analysis")
                
                try:
                    stock_info = yf.Ticker(selected_stock).info
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("#### Valuation Metrics")
                        metrics_html = "<div class='custom-card'>"
                        if 'marketCap' in stock_info:
                            metrics_html += f"<p><span style='color: #cbd5e1;'>Market Cap:</span> <b>₹{stock_info['marketCap']/10000000:.2f} Cr</b></p>"
                        if 'trailingPE' in stock_info:
                            metrics_html += f"<p><span style='color: #cbd5e1;'>P/E Ratio:</span> <b>{stock_info['trailingPE']:.2f}</b></p>"
                        if 'priceToBook' in stock_info:
                            metrics_html += f"<p><span style='color: #cbd5e1;'>P/B Ratio:</span> <b>{stock_info['priceToBook']:.2f}</b></p>"
                        metrics_html += "</div>"
                        st.markdown(metrics_html, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### Financial Metrics")
                        metrics_html = "<div class='custom-card'>"
                        if 'totalRevenue' in stock_info:
                            metrics_html += f"<p><span style='color: #cbd5e1;'>Revenue:</span> <b>₹{stock_info['totalRevenue']/10000000:.2f} Cr</b></p>"
                        if 'profitMargins' in stock_info:
                            metrics_html += f"<p><span style='color: #cbd5e1;'>Profit Margin:</span> <b>{stock_info['profitMargins']:.2%}</b></p>"
                        if 'returnOnEquity' in stock_info:
                            metrics_html += f"<p><span style='color: #cbd5e1;'>ROE:</span> <b>{stock_info['returnOnEquity']:.2%}</b></p>"
                        metrics_html += "</div>"
                        st.markdown(metrics_html, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("#### Dividend & Yield")
                        metrics_html = "<div class='custom-card'>"
                        if 'dividendYield' in stock_info and stock_info['dividendYield']:
                            metrics_html += f"<p><span style='color: #cbd5e1;'>Dividend Yield:</span> <b>{stock_info['dividendYield']:.2%}</b></p>"
                        if 'payoutRatio' in stock_info:
                            metrics_html += f"<p><span style='color: #cbd5e1;'>Payout Ratio:</span> <b>{stock_info['payoutRatio']:.2%}</b></p>"
                        if 'beta' in stock_info:
                            metrics_html += f"<p><span style='color: #cbd5e1;'>Beta:</span> <b>{stock_info['beta']:.2f}</b></p>"
                        metrics_html += "</div>"
                        st.markdown(metrics_html, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if 'longBusinessSummary' in stock_info:
                        st.markdown("#### Business Summary")
                        st.markdown(f"""
                            <div class='custom-card'>
                                <p style='color: #cbd5e1; line-height: 1.6;'>{stock_info['longBusinessSummary'][:600]}...</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f" Error fetching fundamental data: {e}")
            
            with tab5:
                st.markdown("### Candlestick Chart Image Analyzer")
                st.info(
                    "Upload a candlestick chart image to detect patterns (Doji, Hammer, Shooting Star, Engulfing), "
                    "estimate trend, and get AI-powered trading suggestions."
                )
                
                uploaded = st.file_uploader(
                    "Choose an image file",
                    type=["png","jpg","jpeg"],
                    key="image_upload"
                )
                
                if uploaded:
                    pil = Image.open(uploaded).convert("RGB")
                    img_bgr = to_np(pil)
                    img_bgr = resize_keep_aspect(img_bgr, max_width=1200)

                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("#### Original Image")
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

                    with st.spinner(" Analyzing candlestick patterns..."):
                        candles = detect_candles(img_bgr)
                        analysis = analyze(candles)
                        annotated = annotate_image(img_bgr, candles, analysis)

                    with col_b:
                        st.markdown("####  Analyzed Image")
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

                    st.markdown("---")
                    
                    # Analysis Results
                    result_col1, result_col2 = st.columns([1, 2])
                    
                    with result_col1:
                        st.markdown("#### Trading Recommendation")
                        
                        suggestion_colors = {
                            'BUY': 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                            'SELL': 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
                            'HOLD': 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
                        }
                        
                        sug_gradient = suggestion_colors.get(analysis['suggestion'], 'linear-gradient(135deg, #64748b 0%, #475569 100%)')
                        
                        st.markdown(
                            f"""
                            <div style="padding: 2rem; border-radius: 15px; background: {sug_gradient}; 
                                        color: white; text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.15);">
                                <h2 style='margin: 0; color: white; font-size: 3rem;'>{analysis['suggestion']}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        st.markdown(f"""
                            <div class='custom-card'>
                                <p><span style='color: #cbd5e1;'>Detected Trend:</span> 
                                <b style='color: #f1f5f9;'>{analysis['trend'].upper()}</b></p>
                                <p><span style='color: #cbd5e1;'>Trend Slope:</span> 
                                <b style='color: #f1f5f9;'>{analysis['slope']:.2f}</b></p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col2:
                        st.markdown("####  Analysis Details")
                        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                        st.markdown("**Reasoning:**")
                        for reason in analysis['reasons']:
                            st.markdown(f"• {reason}")
                        
                        if analysis['engulfing']:
                            st.markdown("")
                            st.markdown("**Engulfing Patterns Found:**")
                            for e in analysis['engulfing']:
                                st.markdown(f"• Candle {e['index']}: {e['desc']}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Detected Candles Table
                    if not candles:
                        st.warning("⚠️ No candles detected. Try a cleaner chart image or crop out legends/axes.")
                    else:
                        st.markdown(f"####  Detected Candles ({len(candles)} total)")
                        
                        candle_rows = []
                        for i, c in enumerate(candles):
                            pattern = next((r['kind'] for r in analysis['pattern_results'] if r['index'] == i and r['kind']), '-')
                            candle_rows.append({
                                "Index": i,
                                "Color": c['color'].capitalize(),
                                "Body Height (px)": candle_body_height(c),
                                "Total Range (px)": candle_total_range(c),
                                "Pattern": pattern if pattern else '-'
                            })
                        
                        candle_df = pd.DataFrame(candle_rows)
                        st.dataframe(candle_df, use_container_width=True, hide_index=True)
                    
                    st.info(
                        " **Tip:** For best results, crop the image to show only the chart area "
                        "(remove legends, titles, and axis labels). Use clear candle colors (green/red)."
                    )
                    
                else:
                    st.markdown("""
                        <div style='text-align: center; padding: 3rem; background: #1e293b; 
                                    border-radius: 15px; border: 2px dashed #475569;'>
                            <h3 style='color: #cbd5e1;'> Upload a candlestick chart image to begin analysis</h3>
                            <p style='color: #94a3b8;'>Supported formats: PNG, JPG, JPEG</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        else:
            st.error(" Failed to fetch stock data. Please try again or select a different stock.")

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: #94a3b8; padding: 2rem 0; border-top: 2px solid #475569;'>
            <p> Stock Analysis Dashboard | Built with Streamlit & Python</p>
            <p style='font-size: 0.85rem;'>Data provided by Yahoo Finance | For educational purposes only</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    