from datetime import datetime, timezone
from typing import List
from uuid import UUID
import pytz
from apscheduler.triggers.cron import CronTrigger
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED

import pandas as pd
from db.settings import SYMBOL
from helpers import add_bollinger_bands, add_ema, add_macd, add_rsi, connection_is_good, get_latest_candles
import runtime_settings
from services.core.models import DataRun, TickData, BTCFDUSDTick

client = runtime_settings.read_only_client

TICKS_TABLE_NAME = BTCFDUSDTick._meta.db_table
DATA_TABLE_NAME = TickData._meta.db_table
BASE_ASSET = runtime_settings.BASE_ASSET
QUOTE_ASSET = runtime_settings.QUOTE_ASSET
SYMBOL = f"{BASE_ASSET}{QUOTE_ASSET}"
DATA_FREQUENCY_SECS = runtime_settings.DATA_FREQUENCY_SECS


class DataRepository():

    def __init__(self):
        data_run = DataRun()
        data_run.com_or_us = runtime_settings.COM_OR_US
        data_run.is_testnet = runtime_settings.IS_TESTNET
        data_run.base_asset = runtime_settings.BASE_ASSET
        data_run.quote_asset = runtime_settings.QUOTE_ASSET
        data_run.save()
        self.data_run = data_run
    

    def start_data_collection(self, callback=None):
        scheduler = BackgroundScheduler(timezone=pytz.UTC)
        scheduler.add_job(
            self.single_poll_and_store,
            trigger=CronTrigger(second='*/5'),
            id='single_poll_and_store',
            max_instances=1,
            coalesce=False,
        )
        if callback is not None: scheduler.add_listener(callback=callback, mask=EVENT_JOB_EXECUTED)
        scheduler.start()


    def get_tick_data(self) -> TickData:
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

        tick = TickData(
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
        tick.data_run = self.data_run
        return tick
    

    def single_poll_and_store(self):
        if connection_is_good(client=client, wait_until_true=False):
            try:
                tick = self.get_tick_data()
                self.insert_tick_data(tick)
                print(f'stored tick: {tick.timestamp} - {tick.price}')
            except BaseException as e:
                now = datetime.now(timezone.utc)
                print(f'stored tick: {now} - {e}')
                self.insert_empty_tick_data(timestamp=now)
        else:
            now = datetime.now(timezone.utc)
            self.insert_empty_tick_data(timestamp=now)

    
    def insert_tick_data(self, tick: TickData):
        tick.save()


    def insert_empty_tick_data(self, timestamp: datetime):
        TickData.objects.create(
            timestamp=timestamp,
            data_run=self.data_run
        )


    def get_latest_tick_window_data(self, num_ticks: int) -> List[TickData]:
        """
        Fetch the last `num_ticks` tick data from the TickData table.
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - pd.Timedelta(seconds=num_ticks * DATA_FREQUENCY_SECS)
        ticks = TickData.objects.filter(timestamp__gte=start_time, timestamp__lte=end_time).order_by('timestamp')
        return list(ticks)
    

    def get_latest_tick_window_data_by_run(self, num_ticks: int, run_id: UUID) -> List[TickData]:
        """
        Fetch the last `num_ticks` tick data from the TickData table.
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - pd.Timedelta(seconds=num_ticks * DATA_FREQUENCY_SECS)
        ticks = TickData.objects.filter(timestamp__gte=start_time, timestamp__lte=end_time, data_run_id=run_id).order_by('timestamp')
        return list(ticks)

