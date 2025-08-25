from datetime import datetime, timezone
from typing import List
from uuid import UUID
import pytz
from apscheduler.triggers.cron import CronTrigger
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED

import pandas as pd
from db.settings import SYMBOL
from helpers import connection_is_good, get_latest_candles
import runtime_settings
from services.core.models import EMA, MACD, RSI, BollingerBands, DataRun, FeatureSet, TickData
import pandas_ta as ta

client = runtime_settings.read_only_client

DATA_TABLE_NAME = TickData._meta.db_table
BASE_ASSET = runtime_settings.BASE_ASSET
QUOTE_ASSET = runtime_settings.QUOTE_ASSET
SYMBOL = f"{BASE_ASSET}{QUOTE_ASSET}"
DATA_FREQUENCY_SECS = runtime_settings.DATA_FREQUENCY_SECS


class DataRepository():

    def __init__(self, feature_set_name: str = None):
        feature_set = FeatureSet.objects.filter(name=feature_set_name).first()
        if not feature_set:
            feature_set = FeatureSet.objects.filter(name='default').first()
        self.feature_set = feature_set
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
    

    def get_tick_data(self):
        tick = TickData()
        tick.timestamp = datetime.now(timezone.utc)
        ticker = runtime_settings.read_only_client.get_symbol_ticker(symbol=SYMBOL)
        tick.price = float(ticker['price'])

        fs = self.feature_set
        base_set = fs.base_observation_set
        df: pd.DataFrame = get_latest_candles(runtime_settings.read_only_client, SYMBOL, fs.window_length * 50, base_set.candle_interval)
        
        rsi: RSI
        for i, rsi in enumerate(base_set.rsis.order_by('length').all(), start=1):
            setattr(tick, f'rsi_{i}', ta.rsi(df['Close'], length=rsi.length).iloc[-1])

        ema: EMA
        for i, ema in enumerate(base_set.emas.order_by('length').all(), start=1):
            setattr(tick, f'ema_{i}', ta.ema(df['Close'], length=ema.length).iloc[-1])

        macd: MACD
        for macd in base_set.macds.all():
            _macd = ta.macd(close=df['Close'], fast=macd.fast, slow=macd.slow, signal=macd.signal)
            tick.macd_line = _macd[f'MACD_{macd.fast}_{macd.slow}_{macd.signal}'].iloc[-1]
            tick.macd_hist = _macd[f'MACDh_{macd.fast}_{macd.slow}_{macd.signal}'].iloc[-1]
            tick.macd_signal = _macd[f'MACDs_{macd.fast}_{macd.slow}_{macd.signal}'].iloc[-1]

        bband: BollingerBands
        for bband in base_set.bbands.all():
            _bband = ta.bbands(df['Close'], length=bband.length, std=bband.std_dev)
            tick.bb_upper = _bband[f'BBU_{bband.length}_{bband.std_dev}'].iloc[-1]
            tick.bb_middle = _bband[f'BBM_{bband.length}_{bband.std_dev}'].iloc[-1]
            tick.bb_lower = _bband[f'BBL_{bband.length}_{bband.std_dev}'].iloc[-1]

        tick.data_run = self.data_run
        return tick
    

    def single_poll_and_store(self):
        if connection_is_good(client=client, wait_until_true=False):
            try:
                tick = self.get_tick_data()
                self.insert_tick_data(tick)
                print(f'stored tick: {tick.timestamp} - {tick.price}')
            except BaseException as e:
                print(f'tick not stored: {e}')

    
    def insert_tick_data(self, tick: TickData):
        tick.save()


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

