import numpy as np
import pandas as pd


class DerivedFeatureMethods:
    
    @staticmethod
    def all_trend_up(df:pd.DataFrame):
        short_ema = df['ema_1'].iloc[-1]
        mid_ema   = df['ema_2'].iloc[-1]
        long_ema  = df['ema_3'].iloc[-1]
        return int(short_ema > mid_ema > long_ema)
    
    @staticmethod
    def short_gt_mid(df: pd.DataFrame):
        short_ema = df['ema_1'].iloc[-1]
        mid_ema   = df['ema_2'].iloc[-1]
        return int(short_ema > mid_ema)

    @staticmethod
    def mid_gt_long(df: pd.DataFrame):
        mid_ema   = df['ema_2'].iloc[-1]
        long_ema  = df['ema_3'].iloc[-1]
        return int(mid_ema > long_ema)

    @staticmethod
    def all_trend_up(df: pd.DataFrame):
        short_ema = df['ema_1'].iloc[-1]
        mid_ema   = df['ema_2'].iloc[-1]
        long_ema  = df['ema_3'].iloc[-1]
        return int(short_ema > mid_ema > long_ema)

    @staticmethod
    def price_gt_long(df: pd.DataFrame):
        price    = df['price'].iloc[-1]
        long_ema = df['ema_3'].iloc[-1]
        return int(price > long_ema)

    @staticmethod
    def breakout_high(df: pd.DataFrame):
        price = df['price'].iloc[-1]
        return int(price >= df['price'].max())

    @staticmethod
    def bb_squeeze(df: pd.DataFrame):
        upper_bb  = df['bb_upper'].iloc[-1]
        lower_bb  = df['bb_lower'].iloc[-1]
        middle_bb = df['bb_middle'].iloc[-1]
        return float((upper_bb - lower_bb) / middle_bb if middle_bb != 0 else 0)

    @staticmethod
    def ema_slope_sign(df: pd.DataFrame):
        short_ema_now  = df['ema_1'].iloc[-1]
        short_ema_prev = df['ema_1'].iloc[-2]
        return np.sign(short_ema_now - short_ema_prev)

    @staticmethod
    def dist_from_short_ema(df: pd.DataFrame):
        price     = df['price'].iloc[-1]
        short_ema = df['ema_1'].iloc[-1]
        return (price - short_ema) / short_ema if short_ema != 0 else 0

    @staticmethod
    def price_above_upper_bb(df: pd.DataFrame):
        price    = df['price'].iloc[-1]
        upper_bb = df['bb_upper'].iloc[-1]
        return int(price > upper_bb)

    @staticmethod
    def price_below_lower_bb(df: pd.DataFrame):
        price    = df['price'].iloc[-1]
        lower_bb = df['bb_lower'].iloc[-1]
        return int(price < lower_bb)

    @staticmethod
    def price_vs_middle_bb(df: pd.DataFrame):
        price     = df['price'].iloc[-1]
        middle_bb = df['bb_middle'].iloc[-1]
        return (price - middle_bb) / middle_bb if middle_bb != 0 else 0

    @staticmethod
    def bb_width(df: pd.DataFrame):
        upper_bb = df['bb_upper'].iloc[-1]
        lower_bb = df['bb_lower'].iloc[-1]
        return float(upper_bb - lower_bb)

    @staticmethod
    def bb_percent_b(df: pd.DataFrame):
        price    = df['price'].iloc[-1]
        upper_bb = df['bb_upper'].iloc[-1]
        lower_bb = df['bb_lower'].iloc[-1]
        return ((price - lower_bb) / (upper_bb - lower_bb)) if (upper_bb - lower_bb) != 0 else 0
    
    @staticmethod
    def position_exists(position, entry_price, price):
        return float(position)

    @staticmethod
    def position_relative_entry_price(position, entry_price, price):
        return (entry_price / price) if position == 1 else 0.0

    @staticmethod
    def position_unrealized_pnl(position, entry_price, price):
        return ((price - entry_price) / entry_price) if position == 1 else 0.0
    
