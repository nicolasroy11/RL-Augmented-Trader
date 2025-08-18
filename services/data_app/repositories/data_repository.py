from datetime import datetime, timezone
from typing import List

import pandas as pd
from db.data_store import Tick
from db.settings import SYMBOL
from helpers import add_bollinger_bands, add_ema, add_macd, add_rsi, connection_is_good, get_latest_candles
from services.core.models import BTCFDUSDData, BTCFDUSDTick
from db.data_store import DataStore
from django.conf import settings
from binance.client import Client


data_store: DataStore = settings.READ_ONLY_DUCKDB_CONN
TICKS_TABLE_NAME = BTCFDUSDTick._meta.db_table
DATA_TABLE_NAME = BTCFDUSDData._meta.db_table
client = Client(tld='com')
store_frequency_secs = 5


class DataRepository():

    def get_ticks_from_duckdb(self) -> List[Tick]:
        dict_ticks = data_store.get_all_tick_data().to_dict(orient='records')
        tick_list: List[Tick] = [
            Tick(
                timestamp=d["timestamp"],
                price=d["price"],
                rsi_5=d["rsi_5"],
                rsi_7=d["rsi_7"],
                rsi_12=d["rsi_12"],
                ema_short=d["ema_short"],
                ema_mid=d["ema_mid"],
                ema_long=d["ema_long"],
                ema_xlong=d["ema_xlong"],
                macd_line=d["macd_line"],
                macd_hist=d["macd_hist"],
                macd_signal=d["macd_signal"],
                bb_upper=d["bb_upper"],
                bb_middle=d["bb_middle"],
                bb_lower=d["bb_lower"],
                bb_width=d["bb_width"],
                percent_b=d["percent_b"],
                z_bb=d["z_bb"]
            )
            for d in dict_ticks
        ]
        return tick_list


    def save_ticks_to_pg(self) -> List[BTCFDUSDTick]:
        """
        Save a list of Tick dataclass objects into the BTCFDUSDTick table.
        """
        tick_list = self.get_ticks_from_duckdb()
        ticks = [
            BTCFDUSDTick(
                timestamp=t.timestamp.replace(tzinfo=timezone.utc),
                price=t.price,
                rsi_5=t.rsi_5,
                rsi_7=t.rsi_7,
                rsi_12=t.rsi_12,
                ema_short=t.ema_short,
                ema_mid=t.ema_mid,
                ema_long=t.ema_long,
                ema_xlong=t.ema_xlong,
                macd_line=t.macd_line,
                macd_hist=t.macd_hist,
                macd_signal=t.macd_signal,
                bb_upper=t.bb_upper,
                bb_middle=t.bb_middle,
                bb_lower=t.bb_lower
            )
            for t in tick_list
        ]

        BTCFDUSDTick.objects.bulk_create(ticks, ignore_conflicts=False)
        return ticks
    

    def get_tick_data(self) -> BTCFDUSDData:
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

        tick = BTCFDUSDData(
                        timestamp = ts,
                        price = price,
                        rsi_5 = data[f'rsi_{short_rsi_length}'].iloc[-1],
                        rsi_7 = data[f'rsi_{mid_rsi_length}'].iloc[-1],
                        rsi_12 = data[f'rsi_{long_rsi_length}'].iloc[-1],
                        ema_short = data[f'ema_{short_ema_length}'].iloc[-1],
                        ema_mid = data[f'ema_{mid_ema_length}'].iloc[-1],
                        ema_long = data[f'ema_{long_ema_length}'].iloc[-1],
                        ema_xlong = data[f'ema_{xlong_ema_length}'].iloc[-1],
                        macd_line = data['macd_line'].iloc[-1],
                        macd_hist = data['macd_hist'].iloc[-1],
                        macd_signal = data['macd_signal'].iloc[-1],
                        bb_upper = data['bb_upper'].iloc[-1],
                        bb_middle = data['bb_middle'].iloc[-1],
                        bb_lower = data['bb_lower'].iloc[-1],
                    )
        return tick
    

    def single_poll_and_store(self):
        n = 0
        if connection_is_good(client=client, wait_until_true=False):
            try:
                tick = self.get_tick_data()
                self.insert_tick_data(tick)
                print(f'stored tick {n}: {tick.timestamp} - {tick.price}')
            except BaseException as e:
                now = datetime.now(timezone.utc)
                print(f'stored tick {n}: {tick.timestamp} - {e}')
                self.insert_empty_tick_data(timestamp=now)
        else:
            now = datetime.now(timezone.utc)
            self.insert_empty_tick_data(timestamp=now)

    
    def insert_tick_data(self, tick: BTCFDUSDData):
        BTCFDUSDData.objects.create(
            timestamp=tick.timestamp,
            price=tick.price,
            rsi_5=tick.rsi_5,
            rsi_7=tick.rsi_7,
            rsi_12=tick.rsi_12,
            ema_short=tick.ema_short,
            ema_mid=tick.ema_mid,
            ema_long=tick.ema_long,
            ema_xlong=tick.ema_xlong,
            macd_line=tick.macd_line,
            macd_hist=tick.macd_hist,
            macd_signal=tick.macd_signal,
            bb_upper=tick.bb_upper,
            bb_middle=tick.bb_middle,
            bb_lower=tick.bb_lower,
        )

    def insert_empty_tick_data(self, timestamp: datetime):
        BTCFDUSDData.objects.create(
            timestamp=timestamp,
            price=None,
            rsi_5=None,
            rsi_7=None,
            rsi_12=None,
            ema_short=None,
            ema_mid=None,
            ema_long=None,
            ema_xlong=None,
            macd_line=None,
            macd_hist=None,
            macd_signal=None,
            bb_upper=None,
            bb_middle=None,
            bb_lower=None
        )
