from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import logging
from datetime import datetime

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App initialization
app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BINANCE_BASE_URL = "https://api.binance.com/api/v3"
DEFAULT_TIMEFRAME = "4h"
DEFAULT_LIMIT = 100

signal_history = {}

# Helper Functions
def fetch_klines(symbol: str, interval: str = DEFAULT_TIMEFRAME, limit: int = DEFAULT_LIMIT):
    url = f"{BINANCE_BASE_URL}/klines"
    params = {
        "symbol": symbol.upper().replace("-", "").replace("/", ""),
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        )
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise

def calculate_indicators(df):
    df = df.copy()
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['sma_50'] = ta.sma(df['close'], length=50)
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bbands['BBU_20_2.0']
    df['bb_middle'] = bbands['BBM_20_2.0']
    df['bb_lower'] = bbands['BBL_20_2.0']
    df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)
    df['mom'] = ta.mom(df['close'], length=10)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    return df

def generate_features(df):
    df = calculate_indicators(df)
    df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['price_sma_20_ratio'] = df['close'] / df['sma_20']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['macd_cross'] = ((df['macd'] > df['macd_signal']).astype(int) - (df['macd'] < df['macd_signal']).astype(int))
    return df.dropna()

def rule_based_signal(df):
    last = df.iloc[-1]
    signal = "Hold"
    confidence = 0.5
    buy_signals = sell_signals = 0
    buy_conf = sell_conf = 0

    if last['rsi'] < 30:
        buy_signals += 1
        buy_conf += (30 - last['rsi']) / 30 * 0.3
    if last['close'] < last['bb_lower']:
        buy_signals += 1
        buy_conf += 0.2
    if df['macd_cross'].iloc[-1] == 1 and df['macd_cross'].iloc[-2] != 1:
        buy_signals += 1
        buy_conf += 0.25
    if last['sma_20_50_cross'] == 1 and df['sma_20_50_cross'].iloc[-2] == 0:
        buy_signals += 1
        buy_conf += 0.25

    if last['rsi'] > 70:
        sell_signals += 1
        sell_conf += (last['rsi'] - 70) / 30 * 0.3
    if last['close'] > last['bb_upper']:
        sell_signals += 1
        sell_conf += 0.2
    if df['macd_cross'].iloc[-1] == -1 and df['macd_cross'].iloc[-2] != -1:
        sell_signals += 1
        sell_conf += 0.25
    if last['sma_20_50_cross'] == 0 and df['sma_20_50_cross'].iloc[-2] == 1:
        sell_signals += 1
        sell_conf += 0.25

    if buy_signals > sell_signals and buy_signals >= 2:
        signal = "Buy"
        confidence = min(0.95, buy_conf)
    elif sell_signals > buy_signals and sell_signals >= 2:
        signal = "Sell"
        confidence = min(0.95, sell_conf)

    indicators = {
        "rsi": round(last['rsi'], 2),
        "macd": round(last['macd'], 4),
        "macd_signal": round(last['macd_signal'], 4),
        "bb_position": round(last['bb_position'], 2),
        "sma_20": round(last['sma_20'], 2),
        "sma_50": round(last['sma_50'], 2),
        "price": round(last['close'], 2)
    }

    return signal, confidence, indicators

def store_signal(symbol, data):
    if symbol not in signal_history:
        signal_history[symbol] = []
    if len(signal_history[symbol]) >= 100:
        signal_history[symbol].pop(0)
    signal_history[symbol].append(data)

# Routes
@app.get("/")
def root():
    return {"message": "Crypto Signal FastAPI is running"}

@app.get("/signal/{symbol}")
def get_signal(symbol: str, interval: str = DEFAULT_TIMEFRAME):
    df = fetch_klines(symbol, interval)
    df_feat = generate_features(df)
    signal, confidence, indicators = rule_based_signal(df_feat)
    current_price = df['close'].iloc[-1]
    response = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "signal": signal,
        "confidence": round(confidence, 2),
        "price": current_price,
        "indicators": indicators
    }
    store_signal(symbol, response)
    return response

@app.get("/history/{symbol}")
def get_history(symbol: str):
    return {"signals": signal_history.get(symbol, [])}

@app.get("/symbols")
def get_symbols():
    try:
        res = requests.get(f"{BINANCE_BASE_URL}/exchangeInfo")
        res.raise_for_status()
        data = res.json()
        symbols = [
            {
                "symbol": s["symbol"],
                "baseAsset": s["baseAsset"],
                "quoteAsset": s["quoteAsset"]
            }
            for s in data["symbols"]
            if s["status"] == "TRADING" and s["quoteAsset"] in ["USDT", "BTC", "ETH"]
        ]
        return {"symbols": symbols[:100]}
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        return {"error": str(e)}
