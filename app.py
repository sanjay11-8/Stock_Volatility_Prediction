# app.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Candlestick Image Analyzer")

st.title("Candlestick Chart Image Analyzer — OpenCV rule-based")
st.markdown(
    "Upload a candlestick chart image — the app will try to detect candles, identify patterns (Doji, Hammer, Shooting Star, Engulfing), "
    "estimate a short trend and give a Buy/Sell/Hold suggestion."
)

# ---------- Utility functions ----------
def to_np(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def ensure_bgr(img):
    # accept PIL Image or numpy BGR
    if isinstance(img, Image.Image):
        return to_np(img)
    return img

def resize_keep_aspect(img, max_width=1200):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

# ---------- Candle detection ----------
def detect_candles(img_bgr):
    """
    Returns list of candles with fields:
    { 'x': x_center, 'w': width, 'body_top': y, 'body_bottom': y, 'high': y, 'low': y, 'color': 'bull'/'bear' }
    y is pixel row (0 top).
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]

    # Preprocess to detect vertical structures (candles)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to get strong vertical lines
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 10)
    # Morphological operations to connect vertical parts (wicks) and bodies
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    opened = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_v, iterations=1)
    # Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        # Filter out very wide boxes (likely axis, labels, legend)
        if cw > 0.2 * w:
            continue
        # Accept fairly tall items
        if ch < 10:
            continue
        candidate_boxes.append((x, y, cw, ch))

    # Sort left to right by x
    candidate_boxes = sorted(candidate_boxes, key=lambda b: b[0])

    candles = []
    for (x, y, cw, ch) in candidate_boxes:
        # Expand box slightly to include body area
        pad_x = max(2, cw // 8)
        pad_y = max(2, ch // 8)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + cw + pad_x)
        y1 = min(h, y + ch + pad_y)
        box = img[y0:y1, x0:x1]

        # Color detection for body: search column-wise for largest colored block (body)
        hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
        # Detect green-ish and red-ish pixels
        # green range
        mask_g = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
        # red ranges (two segments)
        mask_r1 = cv2.inRange(hsv, (0, 40, 40), (10, 255, 255))
        mask_r2 = cv2.inRange(hsv, (160, 40, 40), (179, 255, 255))
        mask_r = cv2.bitwise_or(mask_r1, mask_r2)

        # If chart uses neutral color (black/white), rely on brightness differences:
        mask_color = cv2.bitwise_or(mask_g, mask_r)
        # fallback if color masks are empty: use bright/dark segmentation
        if cv2.countNonZero(mask_color) < 3:
            # threshold on value channel
            v = hsv[:, :, 2]
            _, mask_bright = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)
            _, mask_dark = cv2.threshold(v, 50, 255, cv2.THRESH_BINARY_INV)
            # Create pseudo masks
            mask_g = mask_bright
            mask_r = mask_dark

        # Determine largest connected vertical colored region -> approximate body area
        combined = cv2.bitwise_or(mask_g, mask_r)
        # find contours inside box
        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        body_top, body_bottom = None, None
        if cnts:
            # pick largest by area
            c = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c)
            # coordinates in full image
            body_top = y0 + by
            body_bottom = y0 + by + bh
            # decide color by which mask has more pixels in body rect
            body_mask = np.zeros_like(combined)
            cv2.drawContours(body_mask, [c], -1, 255, -1)
            g_count = int(cv2.countNonZero(cv2.bitwise_and(body_mask, mask_g)))
            r_count = int(cv2.countNonZero(cv2.bitwise_and(body_mask, mask_r)))
            color = 'bull' if g_count >= r_count else 'bear'
        else:
            # fallback: approximate body as the central area of the bounding box
            body_top = y0 + int((y1 - y0) * 0.35)
            body_bottom = y0 + int((y1 - y0) * 0.65)
            color = 'bull'  # arbitrary fallback

        # For wick high and low: scan vertical in center column to find the topmost and bottommost non-background
        cx = x0 + (x1 - x0) // 2
        column = gray[:, cx]
        # Consider a pixel non-background if darkness is significant (color lines are darker)
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

    # Filter out candles that overlap heavily (keep distinct centers)
    filtered = []
    last_x = -9999
    for c in candles:
        if c['x'] - last_x < max(4, c['w']//2):
            # skip if too close to previous
            continue
        filtered.append(c)
        last_x = c['x']
    return filtered

# ---------- Pattern detection ----------
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
    lower_wick = 0
    upper_wick = 0
    # Determine if body is top or bottom: for bullish hammer body near top, lower wick long
    if c['color'] == 'bull':
        lower_wick = min(c['body_top'], c['body_bottom']) - c['low']
        upper_wick = c['high'] - max(c['body_top'], c['body_bottom'])
    else:
        # for bear colored hammer (inverted convention), still check lower wick
        lower_wick = min(c['body_top'], c['body_bottom']) - c['low']
        upper_wick = c['high'] - max(c['body_top'], c['body_bottom'])
    # Check lower wick large and body small relative to total
    return (lower_wick >= 2 * body) and (body <= 0.35 * total)

def detect_shooting_star(c):
    total = candle_total_range(c)
    body = candle_body_height(c)
    if total == 0:
        return False
    # shooting star -> large upper wick and small body
    if c['color'] == 'bear':
        # upper wick relative to body
        upper_wick = c['high'] - max(c['body_top'], c['body_bottom'])
    else:
        upper_wick = c['high'] - max(c['body_top'], c['body_bottom'])
    return (upper_wick >= 2 * body) and (body <= 0.35 * total)

def detect_engulfing(prev_c, cur_c):
    """Check if cur completely engulfs prev (opposite color)"""
    prev_top = min(prev_c['body_top'], prev_c['body_bottom'])
    prev_bottom = max(prev_c['body_top'], prev_c['body_bottom'])
    cur_top = min(cur_c['body_top'], cur_c['body_bottom'])
    cur_bottom = max(cur_c['body_top'], cur_c['body_bottom'])
    # opposite colors
    if prev_c['color'] == cur_c['color']:
        return None
    # engulfing if cur body covers prev body's vertical span
    if (cur_top <= prev_top) and (cur_bottom >= prev_bottom) and (abs(cur_bottom - cur_top) > abs(prev_bottom - prev_top)):
        return 'bullish' if cur_c['color'] == 'bull' else 'bearish'
    return None

# ---------- Trend detection (approx) ----------
def estimate_close(c):
    # Pixel close value: for bull (green), close is top of body; for bear (red), close is bottom of body
    if c['color'] == 'bull':
        return min(c['body_top'], c['body_bottom'])
    else:
        return max(c['body_top'], c['body_bottom'])

def trend_from_closes(closes):
    if len(closes) < 3:
        return 'sideways', 0.0
    # Simple linear fit
    xs = np.arange(len(closes))
    coeffs = np.polyfit(xs, closes, 1)
    slope = coeffs[0]
    if slope < -0.5:
        return 'uptrend', slope
    elif slope > 0.5:
        return 'downtrend', slope
    else:
        return 'sideways', slope

# ---------- Analysis & recommendation ----------
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

    # Short-term trend using last N closes
    closes = [estimate_close(c) for c in candles]
    trend, slope = trend_from_closes(closes[-8:])  # last 8 candles
    suggestion = 'HOLD'
    reasons = []

    # If any bullish pattern and trend is down or sideways -> BUY
    bullish_patterns = [r for r in results if r['kind'] in ('Hammer',) ]
    bearish_patterns = [r for r in results if r['kind'] in ('Shooting Star',) ]
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

# ---------- Visualization helpers ----------
def annotate_image(img_bgr, candles, analysis):
    img = img_bgr.copy()
    h, w = img.shape[:2]
    # draw candles
    for i, c in enumerate(candles):
        color = (0,255,0) if c['color']=='bull' else (0,0,255)
        # body rect
        cv2.rectangle(img, (c['x0'], c['body_top']), (c['x1'], c['body_bottom']), color, 2)
        # wick
        cv2.line(img, (c['x'], c['high']), (c['x'], c['low']), (100,100,100), 1)
        # index label
        cv2.putText(img, str(i), (c['x']-8, c['body_top']-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # annotate detected patterns
    for r in analysis['pattern_results']:
        if r['kind']:
            c = candles[r['index']]
            cv2.putText(img, r['kind'], (c['x']-30, c['low']+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

    for e in analysis['engulfing']:
        c = candles[e['index']]
        cv2.putText(img, e['type'] + ' Engulf', (c['x']-40, c['high']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 2)

    # add summary text box
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

# ---------- Streamlit UI ----------
uploaded = st.file_uploader("Upload candlestick chart image (PNG/JPG)", type=["png","jpg","jpeg"])
if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    img_bgr = to_np(pil)
    img_bgr = resize_keep_aspect(img_bgr, max_width=1200)

    st.subheader("Input Image")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

    with st.spinner("Detecting candles and analyzing..."):
        candles = detect_candles(img_bgr)
        analysis = analyze(candles)
        annotated = annotate_image(img_bgr, candles, analysis)

    st.subheader("Annotated result")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

    # show details
    st.subheader("Detected candles (left → right)")
    if not candles:
        st.warning("No candles detected. Try a cleaner chart image, crop out legends/axes, or ensure candle colors are clear (green/red).")
    else:
        rows = []
        for i,c in enumerate(candles):
            rows.append({
                "index": i,
                "x": c['x'],
                "width_px": c['w'],
                "body_px": candle_body_height(c),
                "range_px": candle_total_range(c),
                "color": c['color'],
                "high": c['high'],
                "low": c['low']
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows))

    st.subheader("Pattern analysis & recommendation")
    st.write(f"**Trend:** {analysis['trend']} (slope = {analysis['slope']:.2f})")
    st.write(f"**Suggestion:** **{analysis['suggestion']}**")
    st.write("**Reasons / details:**")
    for r in analysis['reasons']:
        st.write("- " + r)
    if analysis['engulfing']:
        st.write("**Engulfing patterns:**")
        for e in analysis['engulfing']:
            st.write(f"- At candle index {e['index']}: {e['desc']}")

    st.markdown("---")
    st.caption("Note: This is a rule-based visual detector using pixel geometry. It is best used as a signal helper. For real trading use combine with numeric OHLC, volume, and risk management.")
else:
    st.info("Upload an image to begin. For best results, crop the image to the chart area (remove legends, title, axis labels) and use clear candle colors (green/red).")
