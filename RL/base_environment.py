import duckdb
import pandas as pd
import numpy as np

class BaseTradingEnv:
    def __init__(self, db_path, window_size=10):
        self.con = duckdb.connect(db_path)
        self.window_size = window_size
        self.data = self.load_data()
        self.current_step = 0

    def load_data(self):
        # Load all tick data sorted by timestamp
        query = "SELECT * FROM ticks ORDER BY timestamp"
        df = self.con.execute(query).fetchdf()
        return df

    def reset(self):
        # Reset the environment to the start
        self.current_step = self.window_size
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        # Return the window of data for the current step
        window = self.data.iloc[self.current_step - self.window_size:self.current_step]
        # Convert to numpy array for easier processing by agents
        return window.drop(columns=['timestamp']).to_numpy()

    def _get_normalized_observation(self):
        """Get normalized price window + derived features"""
        obs_df = self.data.iloc[self.current_step - self.window_size : self.current_step].copy()

        # Get latest prices & indicators
        price = obs_df['price'].iloc[-1]
        short_ema = obs_df['ema_short'].iloc[-1]
        mid_ema   = obs_df['ema_mid'].iloc[-1]
        long_ema  = obs_df['ema_long'].iloc[-1]
        upper_bb  = obs_df['bb_upper'].iloc[-1]
        lower_bb  = obs_df['bb_lower'].iloc[-1]
        middle_bb = obs_df['bb_middle'].iloc[-1]

        # === Derived Features ===
        derived_features = {
            "short_gt_mid": int(short_ema > mid_ema),
            "mid_gt_long": int(mid_ema > long_ema),
            "all_trend_up": int(short_ema > mid_ema > long_ema),
            "price_gt_long": int(price > long_ema),
            "breakout_high": int(price >= obs_df['price'].max()),
            "bb_squeeze": float((upper_bb - lower_bb) / middle_bb if middle_bb != 0 else 0),
            "ema_slope_sign": np.sign(short_ema - obs_df['ema_short'].iloc[-2]),
            "dist_from_short_ema": (price - short_ema) / short_ema if short_ema != 0 else 0
        }

        # Normalize base features (window data)
        base_features = obs_df.drop(columns=['timestamp']).to_numpy().astype(np.float32)
        base_mean = base_features.mean(axis=0)
        base_std = base_features.std(axis=0) + 1e-8  # avoid div by zero
        base_features_norm = ((base_features - base_mean) / base_std).flatten()

        # Normalize derived features separately
        derived_array = np.array(list(derived_features.values()), dtype=np.float32)
        derived_mean = derived_array.mean()
        derived_std = derived_array.std() + 1e-8
        derived_features_norm = (derived_array - derived_mean) / derived_std

        # Concatenate normalized base + normalized derived
        obs_vector = np.concatenate([base_features_norm, derived_features_norm])
        return obs_vector

    def normalized_reset(self):
        # Reset the environment to the start
        self.current_step = self.window_size
        return self._get_normalized_observation()

    def step(self, action):
        # For now, action is not used, reward is zero
        self.current_step += 1
        done = self.current_step >= len(self.data)
        obs = self._get_observation() if not done else None
        reward = 0.0  # placeholder
        info = {}
        return obs, reward, done, info
