from dataclasses import dataclass, fields
from typing import List
import uuid
from django.db import models
from binance.enums import SIDE_BUY, SIDE_SELL
import pandas as pd
import numpy as np

import runtime_settings

DUCKDB_TRANSFERS_SCHEMA = 'duckdb_transfers'

WINDOW_LENGTH = runtime_settings.DATA_TICKS_WINDOW_LENGTH


class TrainingFields(models.Model):
    class Meta:
        abstract = True

    price = models.FloatField(null=True)
    rsi_5 = models.FloatField(null=True)
    rsi_7 = models.FloatField(null=True)
    rsi_12 = models.FloatField(null=True)
    ema_short = models.FloatField(null=True)
    ema_mid = models.FloatField(null=True)
    ema_long = models.FloatField(null=True)
    ema_xlong = models.FloatField(null=True)
    macd_line = models.FloatField(null=True)
    macd_hist = models.FloatField(null=True)
    macd_signal = models.FloatField(null=True)
    bb_upper = models.FloatField(null=True)
    bb_middle = models.FloatField(null=True)
    bb_lower = models.FloatField(null=True)
        
    @staticmethod
    def get_fields():
        return {field.name for field in TrainingFields._meta.get_fields() if field.concrete and not field.many_to_many}

    @dataclass
    class ObservationFeatures:

        # === Base features ===
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

        # === Derived features === 
        short_gt_mid: int
        mid_gt_long: int
        all_trend_up: int
        price_gt_long: int
        breakout_high: int
        bb_squeeze: float
        ema_slope_sign: float
        dist_from_short_ema: float
        price_above_upper_bb: int
        price_below_lower_bb: int
        price_vs_middle_bb: float
        bb_width: float
        bb_percent_b: float

        # Position features
        position: float
        relative_entry_price: float
        unrealized_pnl: float

        @classmethod
        def get_fields(cls):
            return [f.name for f in fields(cls)]
        
        @classmethod
        def get_vector_size(cls):
            tfl = len(TrainingFields.get_fields())
            return (tfl * runtime_settings.DATA_TICKS_WINDOW_LENGTH) + (len([f.name for f in fields(cls)]) - tfl)

        @classmethod
        def normalized_vector_from_obs_df(cls, obs_df: pd.DataFrame, position, entry_price):

            # === Derived features ===
            price = obs_df['price'].iloc[-1]
            short_ema = obs_df['ema_short'].iloc[-1]
            mid_ema   = obs_df['ema_mid'].iloc[-1]
            long_ema  = obs_df['ema_long'].iloc[-1]
            upper_bb  = obs_df['bb_upper'].iloc[-1]
            lower_bb  = obs_df['bb_lower'].iloc[-1]
            middle_bb = obs_df['bb_middle'].iloc[-1]
            derived_features = {
                "short_gt_mid": int(short_ema > mid_ema),
                "mid_gt_long": int(mid_ema > long_ema),
                "all_trend_up": int(short_ema > mid_ema > long_ema),
                "price_gt_long": int(price > long_ema),
                "breakout_high": int(price >= obs_df['price'].max()),
                "bb_squeeze": float((upper_bb - lower_bb) / middle_bb if middle_bb != 0 else 0),
                "ema_slope_sign": np.sign(short_ema - obs_df['ema_short'].iloc[-2]),
                "dist_from_short_ema": (price - short_ema) / short_ema if short_ema != 0 else 0,
                "price_above_upper_bb": int(price > upper_bb),
                "price_below_lower_bb": int(price < lower_bb),
                "price_vs_middle_bb": (price - middle_bb) / middle_bb if middle_bb != 0 else 0,
                "bb_width": float(upper_bb - lower_bb),
                "bb_percent_b": ((price - lower_bb) / (upper_bb - lower_bb)) if (upper_bb - lower_bb) != 0 else 0,
            }

            # === Position state features ===
            current_price = price
            position_features = np.array([
                float(position),  # 0 = flat, 1 = long
                (entry_price / current_price) if position == 1 else 0.0,  # relative entry price
                ((current_price - entry_price) / entry_price) if position == 1 else 0.0  # % unrealized pnl
            ], dtype=np.float32)

            # === Normalize base features ===
            base_features = TickData.remove_non_training_fields_in_df(df=obs_df)
            base_features = base_features.to_numpy().astype(np.float32)
            base_mean = base_features.mean(axis=0)
            base_std = base_features.std(axis=0) + 1e-8
            base_features_norm = ((base_features - base_mean) / base_std).flatten()

            # === Normalize derived features ===
            derived_array = np.array(list(derived_features.values()), dtype=np.float32)
            derived_mean = derived_array.mean()
            derived_std = derived_array.std() + 1e-8
            derived_features_norm = (derived_array - derived_mean) / derived_std

            # === Normalize position features (already ratio-based, so just z-score) ===
            pos_mean = position_features.mean()
            pos_std = position_features.std() + 1e-8
            position_features_norm = (position_features - pos_mean) / pos_std

            # === Final observation vector ===
            return np.concatenate([base_features_norm, derived_features_norm, position_features_norm])
                

        def normalized_vector(self):
            arr = np.array([getattr(self, f) for f in self.__dataclass_fields__], dtype=np.float32)
            return (arr - arr.mean()) / (arr.std() + 1e-8)


class BTCFDUSDTick(TrainingFields):
    class Meta:
        db_table = f'"{DUCKDB_TRANSFERS_SCHEMA}"."btcfdusd_ticks"'

    def __str__(self):
        return f"{self.timestamp} | {self.price}"
    
    timestamp = models.DateTimeField(unique=True, db_index=True)


class DataRun(models.Model):
    class Meta:
        db_table = f'"public"."data_runs"'

    def __str__(self):
        return f"{self.id}_{self.com_or_us}_{self.is_testnet}_{self.base_asset}_{self.quote_asset}"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, verbose_name='data_run_id', null=False)
    com_or_us = models.CharField(max_length=3, choices=[('com', 'com'), ('us', 'us')], null=False)
    is_testnet = models.BooleanField(null=False)
    base_asset = models.CharField(max_length=7, null=False)
    quote_asset = models.CharField(max_length=7, null=False)
    window_length = models.IntegerField(null=False, default=runtime_settings.DATA_TICKS_WINDOW_LENGTH)
    run_type = models.CharField(max_length=8, choices=[('training', 'training'), ('trading', 'trading')], null=True)


class TickData(TrainingFields):
    class Meta:
        db_table = f'"public"."tick_data"'

    def __str__(self):
        return f"{self.price}"
    
    @staticmethod
    def remove_non_training_fields_in_df(df: pd.DataFrame):
        """
        Keep only the columns in the DataFrame that match fields of the given Django model.
        """
        model_fields = {field.name for field in TrainingFields._meta.get_fields() if field.concrete and not field.many_to_many}
        valid_columns = [col for col in df.columns if col in model_fields]
        return df[valid_columns]

    @staticmethod
    def list_to_env_dataframe(tick_list: List['TickData']) -> pd.DataFrame:
        ticks_df = pd.DataFrame.from_records([t.__dict__ for t in tick_list])
        ticks_df = ticks_df.drop(columns=["_state"], errors="ignore")
        if len(tick_list) < WINDOW_LENGTH:
            print(f'interpolating {WINDOW_LENGTH - len(tick_list)} ticks')
            ticks_df = TickData.get_interpolated_window_df(ticks_df)
        return ticks_df.reset_index(drop=True)
    
    @staticmethod
    def get_interpolated_window_df(ticks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce a WINDOW_LENGTH DataFrame of TickData at 5-second intervals.
        - Linear interpolation for numeric fields.
        - Handles millisecond timestamps correctly.
        - Ensures exact WINDOW_LENGTH rows via backfilling at start if necessary.
        """

        df = ticks_df.copy().sort_values("timestamp")

        # Determine last timestamp and window start
        last_ts = df["timestamp"].max().ceil("5s")
        first_ts = last_ts - pd.Timedelta(seconds=5 * (WINDOW_LENGTH - 1))

        # Create the fixed 5-second grid
        grid_df = pd.DataFrame({"timestamp": pd.date_range(start=first_ts, end=last_ts, freq="5s")})

        # Merge original data onto the grid using nearest previous timestamps
        merged = pd.merge_asof(grid_df, df, on="timestamp", direction="backward")

        # Interpolate numeric columns linearly
        numeric_cols = merged.select_dtypes(include="number").columns
        merged[numeric_cols] = merged[numeric_cols].interpolate(method="linear").bfill()

        return merged

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, verbose_name='tick_data_id', null=False, db_index=True)
    timestamp = models.DateTimeField(unique=True, db_index=True)
    data_run = models.ForeignKey(DataRun, on_delete=models.CASCADE, null=False)


class TickProbabilities(models.Model):
    class Meta:
        db_table = f'"public"."tick_probabilities"'

    tick_data = models.ForeignKey(TickData, on_delete=models.CASCADE, null=False)
    buy_prob = models.FloatField(null=True)
    hold_prob = models.FloatField(null=True)
    sell_prob = models.FloatField(null=True)


class TradingSession(models.Model):
    class Meta:
        db_table = f'"public"."trading_sessions"'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, verbose_name='trading_session_id', null=False)
    data_run = models.ForeignKey(DataRun, on_delete=models.CASCADE, null=False)
    blocking = models.BooleanField(null=False)


class Transaction(models.Model):
    class Meta:
        db_table = f'"public"."transactions"'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, verbose_name='single_buy_cycle_id', null=False)
    side = models.CharField(max_length=8, choices=[(SIDE_BUY, SIDE_BUY), (SIDE_SELL, SIDE_SELL)], null=True)
    tick_data = models.ForeignKey(TickData, on_delete=models.CASCADE, null=False)
    trading_session = models.ForeignKey(TradingSession, on_delete=models.CASCADE, null=False)
    strike_price = models.FloatField()
