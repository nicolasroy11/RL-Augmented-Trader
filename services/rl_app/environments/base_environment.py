import numpy as np
import pandas as pd
from services.core.models import BTCFDUSDData


class BaseTradingEnvironment:
    def __init__(self, data: pd.DataFrame, window_size=10):
        self.window_size = window_size
        self.data = data
        self.current_step = 0
        self.position: int = 0
        self.entry_price: float = None

    def load_data(self):
        ticks = BTCFDUSDData.objects.all().order_by("timestamp").values()
        df = pd.DataFrame.from_records(ticks)
        return df.reset_index(drop=True)

    def reset(self):
        self.current_step = self.window_size
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        window = self.data.iloc[self.current_step - self.window_size : self.current_step]
        return window.drop(columns=["id", "timestamp"]).to_numpy()

    def _get_normalized_observation(self):
        obs_df = self.data.iloc[self.current_step - self.window_size : self.current_step].copy()

        # === Latest prices & indicators ===
        price = obs_df['price'].iloc[-1]
        short_ema = obs_df['ema_short'].iloc[-1]
        mid_ema   = obs_df['ema_mid'].iloc[-1]
        long_ema  = obs_df['ema_long'].iloc[-1]
        upper_bb  = obs_df['bb_upper'].iloc[-1]
        lower_bb  = obs_df['bb_lower'].iloc[-1]
        middle_bb = obs_df['bb_middle'].iloc[-1]

        # === Derived features ===
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
            float(self.position),  # 0 = flat, 1 = long
            (self.entry_price / current_price) if self.position == 1 else 0.0,  # relative entry price
            ((current_price - self.entry_price) / self.entry_price) if self.position == 1 else 0.0  # % unrealized pnl
        ], dtype=np.float32)

        # === Normalize base features ===
        base_features = obs_df.drop(columns=["id", "timestamp"])
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


    def normalized_reset(self):
        self.current_step = self.window_size
        return self._get_normalized_observation()

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data)
        obs = self._get_observation() if not done else None
        reward = 0.0
        info = {}
        return obs, reward, done, info
