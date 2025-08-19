# binance_testnet_bot.py

import time
import pandas as pd
from datetime import datetime, timezone
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET

import runtime_settings
from services.data_app.repositories.data_repository import DataRepository
data_repo = DataRepository()

# ------------------------------
# Config
# ------------------------------

BASE_ASSET = runtime_settings.BASE_ASSET
QUOTE_ASSET = runtime_settings.QUOTE_ASSET
SYMBOL = f"{BASE_ASSET}{QUOTE_ASSET}"
DATA_FREQUENCY_SECS = runtime_settings.DATA_FREQUENCY_SECS
HISTORY_WINDOW = runtime_settings.DATA_TICKS_WINDOW        # ticks for indicators

API_KEY = runtime_settings.binance_api_key
API_SECRET = runtime_settings.binance_secret_key

BASE_ASSET = runtime_settings.BASE_ASSET
QUOTE_ASSET = runtime_settings.QUOTE_ASSET

# ------------------------------
# Binance client (testnet)
# ------------------------------
client = runtime_settings.write_client

# ------------------------------
# Portfolio tracking
# ------------------------------
class Portfolio:
    def __init__(self):
        self.position = 0.0
        self.entry_price = 0.0
        self.realized_pnl = 0.0

portfolio = Portfolio()

# ------------------------------
# Simple signal -> action
# ------------------------------
def compute_action_probs(tick):
    # Example: simple z-score of price vs ema_150
    z_score = (tick['price'] - tick['ema_150']) / tick['ema_150']
    probs = {'buy': max(0, -z_score), 'sell': max(0, z_score), 'hold': 1 - abs(z_score)}
    total = sum(probs.values())
    for k in probs:
        probs[k] /= total
    return probs

def decide_trade(action_probs):
    trade_signal = max(action_probs, key=action_probs.get)
    qty = 0.001  # Example quantity
    return trade_signal, qty

# ------------------------------
# Execute trade on testnet
# ------------------------------
def execute_trade(symbol, trade_signal, qty, price):
    if trade_signal == "buy":
        order = client.futures_create_order(
            symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quantity=qty
        )
        portfolio.entry_price = price
        portfolio.position += qty
    elif trade_signal == "sell" and portfolio.position > 0:
        order = client.futures_create_order(
            symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=qty
        )
        pnl = (price - portfolio.entry_price) * qty
        portfolio.realized_pnl += pnl
        portfolio.position -= qty
    # hold does nothing

# ------------------------------
# Main loop
# ------------------------------
def run_bot():
    history_df = pd.DataFrame()

    while True:
        tick = data_repo.get_tick_data(SYMBOL)
        history_df = history_df.append(tick, ignore_index=True)
        if len(history_df) > HISTORY_WINDOW:
            history_df = history_df.iloc[-HISTORY_WINDOW:]

        action_probs = compute_action_probs(tick)
        trade_signal, qty = decide_trade(action_probs)
        execute_trade(SYMBOL, trade_signal, qty, tick['price'])

        unrealized_pnl = (tick['price'] - portfolio.entry_price) * portfolio.position
        print(
            f"{tick['timestamp']} | Pos: {portfolio.position:.6f} {BASE_ASSET} @ {portfolio.entry_price:.2f} | "
            f"Unrealized: {unrealized_pnl:.2f} | Realized: {portfolio.realized_pnl:.2f} | "
            f"Action probs: {action_probs}"
        )

        time.sleep(DATA_FREQUENCY_SECS)

# ------------------------------
# Run bot
# ------------------------------
if __name__ == "__main__":
    run_bot()
