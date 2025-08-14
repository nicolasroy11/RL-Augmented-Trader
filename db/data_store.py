from dataclasses import dataclass
from binance.client import Client
import duckdb
from datetime import datetime, timezone
import time
import pandas as pd
from db.settings import DUCKDB_FILE_PATH, SYMBOL, TICKS_TABLE_NAME
from helpers import add_bollinger_bands, add_ema, add_macd, add_rsi, connection_is_good, get_latest_candles


client = Client(tld='com')
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
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    percent_b: float
    z_bb: float


class DataStore:

    def __init__(self, db_path: str, readonly: bool = True):
        self.con = duckdb.connect(db_path, read_only=readonly)


    def create_table(self):
        self.con.execute(f"""
            CREATE TABLE IF NOT EXISTS {TICKS_TABLE_NAME} (
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
                macd_signal DOUBLE,
                bb_upper DOUBLE,
                bb_middle DOUBLE,
                bb_lower DOUBLE,
                bb_width DOUBLE,
                percent_b DOUBLE,
                z_bb DOUBLE
            )
        """)


    def insert_tick_data(self, tick: Tick):
        self.con.execute(f"""
            INSERT INTO {TICKS_TABLE_NAME} (timestamp, price, rsi_5, rsi_7, rsi_12, ema_short, ema_mid, ema_long, ema_xlong, macd_line, macd_hist, macd_signal, bb_upper, bb_middle, bb_lower, bb_width, percent_b, z_bb)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tick.timestamp, tick.price, tick.rsi_5, tick.rsi_7, tick.rsi_12, tick.ema_short, tick.ema_mid, tick.ema_long, tick.ema_xlong, tick.macd_line, tick.macd_hist, tick.macd_signal, tick.bb_upper, tick.bb_middle, tick.bb_lower, tick.bb_width, tick.percent_b, tick.z_bb))


    def insert_empty_tick_data(self, timestamp: datetime):
        self.con.execute(f"""
            INSERT INTO {TICKS_TABLE_NAME} (timestamp, price, rsi_5, rsi_7, rsi_12, ema_short, ema_mid, ema_long, ema_xlong, macd_line, macd_hist, macd_signal, bb_upper, bb_middle, bb_lower, bb_width, percent_b, z_bb)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None))


    def continuous_poll_and_store(self):
        while True:
            t0 = time.time()
            self.single_poll_and_store()
            t1 = time.time()
            time_to_sleep = store_frequency_secs - (t1 - t0)
            if time_to_sleep > 0: time.sleep(time_to_sleep)

    
    def single_poll_and_store(self):
        if connection_is_good(client=client, wait_until_true=False):
            try:
                tick = self.get_tick_data()
                self.insert_tick_data(tick)
                print(f'stored: {tick.timestamp}')
            except BaseException as e:
                now = datetime.now(timezone.utc)
                self.insert_empty_tick_data(timestamp=now)
        else:
            now = datetime.now(timezone.utc)
            self.insert_empty_tick_data(timestamp=now)


    def get_tick_data(self) -> Tick:
        short_ema_length = 7
        mid_ema_length = 12
        long_ema_length = 30
        xlong_ema_length = 150
        short_rsi_length = 5
        mid_rsi_length = 7
        long_rsi_length = 12
        bollinger_length = 14
        ts = datetime.now(timezone.utc)
        ticker = client.get_symbol_ticker(symbol=SYMBOL)
        price = float(ticker['price'])
        data: pd.DataFrame = get_latest_candles(client, SYMBOL, xlong_ema_length * 50, client.KLINE_INTERVAL_1MINUTE)
        data = add_rsi(data, length=short_rsi_length)
        data = add_rsi(data, length=mid_rsi_length)
        data = add_rsi(data, length=long_rsi_length)
        data = add_ema(data, length=short_ema_length)
        data = add_ema(data, length=mid_ema_length)
        data = add_ema(data, length=long_ema_length)
        data = add_ema(data, length=xlong_ema_length)
        data = add_macd(data)
        data = add_bollinger_bands(data, length=bollinger_length, std_dev=2.0)
        tick = Tick(
                        ts,
                        price,
                        data[f'rsi_{short_rsi_length}'].iloc[-1],
                        data[f'rsi_{mid_rsi_length}'].iloc[-1],
                        data[f'rsi_{long_rsi_length}'].iloc[-1],
                        data[f'ema_{short_ema_length}'].iloc[-1],
                        data[f'ema_{mid_ema_length}'].iloc[-1],
                        data[f'ema_{long_ema_length}'].iloc[-1],
                        data[f'ema_{xlong_ema_length}'].iloc[-1],
                        data['macd_line'].iloc[-1],
                        data['macd_hist'].iloc[-1],
                        data['macd_signal'].iloc[-1],
                        data['bb_upper'].iloc[-1],
                        data['bb_middle'].iloc[-1],
                        data['bb_lower'].iloc[-1],
                        data['bb_width'].iloc[-1],
                        data['percent_b'].iloc[-1],
                        data['z_bb'].iloc[-1]
                    )
        return tick
    
    
    def test_print(self):
        df = self.con.execute(f"SELECT * FROM {TICKS_TABLE_NAME} ORDER BY timestamp DESC LIMIT 10").fetchdf()
        print(df)


    def get_recent_tick_data(self, num_ticks: int = 10):
        df = self.con.execute(f"SELECT * FROM {TICKS_TABLE_NAME} ORDER BY timestamp DESC LIMIT {num_ticks}").fetchdf()
        return df


    def get_all_tick_data(self):
        df = self.con.execute(f"SELECT * FROM {TICKS_TABLE_NAME} ORDER BY timestamp ASC").fetchdf()
        return df


if __name__ == "__main__":
    datastore = DataStore(DUCKDB_FILE_PATH, readonly=False)
    datastore.create_table()
    datastore.continuous_poll_and_store()
