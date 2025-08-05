from dataclasses import dataclass
from binance.client import Client
import duckdb
from datetime import datetime, timezone
import time
import pandas_ta as ta
from helpers import connection_is_good, get_latest_candles
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "ticks.duckdb")
con = duckdb.connect(db_path)

client = Client(tld='com')
symbol = 'BTCFDUSD'
store_frequency_secs = 5


@dataclass
class Tick:
    timestamp: datetime
    price: float
    rsi_5: float
    rsi_7: float
    rsi_12: float
    ema_short: float
    ema_mid: float
    ema_long: float
    ema_xlong: float
    macd_line: float
    macd_hist: float
    macd_signal: float


def create_table():
    con.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            timestamp TIMESTAMP,
            price DOUBLE,
            rsi_5 DOUBLE,
            rsi_7 DOUBLE,
            rsi_12 DOUBLE,
            ema_short DOUBLE,
            ema_mid DOUBLE,
            ema_long DOUBLE,
            ema_xlong DOUBLE,
            macd_line DOUBLE,
            macd_hist DOUBLE,
            macd_signal DOUBLE
        )
    """)


def insert_tick_data(tick: Tick):
    con.execute("""
        INSERT INTO ticks (timestamp, price, rsi_5, rsi_7, rsi_12, ema_short, ema_mid, ema_long, ema_xlong, macd_line, macd_hist, macd_signal)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (tick.timestamp, tick.price, tick.rsi_5, tick.rsi_7, tick.rsi_12, tick.ema_short, tick.ema_mid, tick.ema_long, tick.ema_xlong, tick.macd_line, tick.macd_hist, tick.macd_signal))


def insert_empty_tick_data(timestamp: datetime):
    con.execute("""
        INSERT INTO ticks (timestamp, price, rsi_5, rsi_7, rsi_12, ema_short, ema_mid, ema_long, ema_xlong, macd_line, macd_hist, macd_signal)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, None, None, None, None, None, None, None, None, None, None, None))


def poll_and_store():
    while True:
        t0 = time.time()
        if connection_is_good(client=client, wait_until_true=False):
            try:
                tick = get_tick_data()
                insert_tick_data(tick)
                print(f'stored: {tick.timestamp}')
            except BaseException as e:
                now = datetime.now(timezone.utc)
                insert_empty_tick_data(timestamp=now)
        else:
            now = datetime.now(timezone.utc)
            insert_empty_tick_data(timestamp=now)
        t1 = time.time()
        time_to_sleep = store_frequency_secs - (t1 - t0)
        if time_to_sleep > 0: time.sleep(time_to_sleep)


def get_tick_data() -> Tick:
    short_ema_length = 7
    mid_ema_length = 12
    long_ema_length = 30
    xlong_ema_length = 150
    short_rsi_length = 5
    mid_rsi_length = 7
    long_rsi_length = 12
    ts = datetime.now(timezone.utc)
    ticker = client.get_symbol_ticker(symbol=symbol)
    price = float(ticker['price'])
    data = get_latest_candles(client, symbol, xlong_ema_length * 50, client.KLINE_INTERVAL_1MINUTE)
    data['EMA_short'] = ta.ema(length=short_ema_length, close=data['Close'])
    data['EMA_mid'] = ta.ema(length=mid_ema_length, close=data['Close'])
    data['EMA_long'] = ta.ema(length=long_ema_length, close=data['Close'])
    data['EMA_xlong'] = ta.ema(length=xlong_ema_length, close=data['Close'])
    data['RSI_5'] = ta.rsi(length=short_rsi_length, close=data['Close'])
    data['RSI_7'] = ta.rsi(length=mid_rsi_length, close=data['Close'])
    data['RSI_12'] = ta.rsi(length=long_rsi_length, close=data['Close'])
    macd = ta.macd(close=data['Close'])
    macd_line = macd['MACD_12_26_9'].iloc[-1]
    macd_hist = macd['MACDh_12_26_9'].iloc[-1]
    macd_signal = macd['MACDs_12_26_9'].iloc[-1]
    tick = Tick(ts, price, data['RSI_5'].iloc[-1], data['RSI_7'].iloc[-1], data['RSI_12'].iloc[-1], data['EMA_short'].iloc[-1], data['EMA_mid'].iloc[-1], data['EMA_long'].iloc[-1], data['EMA_xlong'].iloc[-1], macd_line, macd_hist, macd_signal)
    return tick


def test_print():
    df = con.execute("SELECT * FROM ticks ORDER BY timestamp DESC LIMIT 10").fetchdf()
    print(df)


def get_recent_tick_data(self, num_ticks: int = 10):
    df = con.execute(f"SELECT * FROM ticks ORDER BY timestamp DESC LIMIT {num_ticks}").fetchdf()
    return df


if __name__ == "__main__":
    create_table()
    poll_and_store()
