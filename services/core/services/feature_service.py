import numpy as np
import pandas as pd
import pandas_ta as ta

import runtime_settings


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
    def ema_1_slope_sign(df: pd.DataFrame):
        short_ema_now  = df['ema_1'].iloc[-1]
        short_ema_prev = df['ema_1'].iloc[-2]
        return np.sign(short_ema_now - short_ema_prev)
    
    @staticmethod
    def ema_2_slope_sign(df: pd.DataFrame):
        short_ema_now  = df['ema_2'].iloc[-1]
        short_ema_prev = df['ema_2'].iloc[-2]
        return np.sign(short_ema_now - short_ema_prev)
    
    @staticmethod
    def ema_3_slope_sign(df: pd.DataFrame):
        short_ema_now  = df['ema_3'].iloc[-1]
        short_ema_prev = df['ema_3'].iloc[-2]
        return np.sign(short_ema_now - short_ema_prev)
    
    @staticmethod
    def ema_4_slope_sign(df: pd.DataFrame):
        short_ema_now  = df['ema_4'].iloc[-1]
        short_ema_prev = df['ema_4'].iloc[-2]
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
    def _round_trip_fee_frac() -> float:
        """
        Round-trip fee fraction = (bps / 1e4).
        E.g., maker+maker = 0.0004 (4 bps), taker+taker = 0.0008 (8 bps).
        """
        fee_bps = runtime_settings.SIM_ROUND_TRIP_FEE_BPS
        return fee_bps / 1e4

    @staticmethod
    def position_relative_entry_price(position, entry_price, price):
        """
        Ratio entry/price, adjusted for fees.
        """
        if position != 1 or not entry_price or not price:
            return 0.0
        return (entry_price / price) - DerivedFeatureMethods._round_trip_fee_frac()

    @staticmethod
    def position_unrealized_pnl(position, entry_price, price):
        """
        Fractional PnL relative to entry, net of round-trip fees.
        """
        if position != 1 or not entry_price or not price:
            return 0.0
        gross = (price - entry_price) / entry_price
        return gross - DerivedFeatureMethods._round_trip_fee_frac()
    
    @staticmethod
    def threshold_rsi_crossed_up_recent(df: pd.DataFrame) -> int:
        if "rsi_1" not in df.columns or len(df) < 2:
            return 0
        rsi = df["rsi_1"]
        thr = pd.Series(30, index=rsi.index)
        crossup = ta.cross(rsi, thr, above=True)
        window = crossup.iloc[-10:] if len(crossup) >= 10 else crossup
        return int(window.max() == 1)

    @staticmethod
    def threshold_rsi_crossed_down_recent(df: pd.DataFrame) -> int:
        if "rsi_1" not in df.columns or len(df) < 2:
            return 0
        rsi = df["rsi_1"]
        thr = pd.Series(70, index=rsi.index)
        crossdown = ta.cross(rsi, thr, below=True)
        window = crossdown.iloc[-10:] if len(crossdown) >= 10 else crossdown
        return int(window.max() == 1)
    
    @staticmethod
    def threshold_price_is_highest_in_window(dataframe: pd.DataFrame) -> int:
        if "price" not in dataframe.columns or len(dataframe) == 0:
            return 0

        window_length = min(180, len(dataframe))    # 5 second intervals * 12 * 5 * 3 = 3 candles of 5 minutes
        current_price = float(dataframe["price"].iloc[-1])
        window_high = float(dataframe["price"].iloc[-window_length:].max())

        tolerance = window_high
        return int(current_price >= window_high - tolerance)
    
    
    
