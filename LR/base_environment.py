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
    
    def normalized_reset(self):
        # Reset the environment to the start
        self.current_step = self.window_size
        return self._get_normalized_observation()

    def _get_normalized_observation(self) -> np.ndarray:
        window = self.data.iloc[self.current_step - self.window_size : self.current_step].copy()
        window = window.drop(columns=["timestamp"], errors="ignore")

        for col in window.columns:
            mean = window[col].mean()
            std = window[col].std()
            if std == 0 or np.isnan(std):
                window[col] = 0.0
            else:
                window[col] = (window[col] - mean) / std

        return window.to_numpy()

    def step(self, action):
        # For now, action is not used, reward is zero
        self.current_step += 1
        done = self.current_step >= len(self.data)
        obs = self._get_observation() if not done else None
        reward = 0.0  # placeholder
        info = {}
        return obs, reward, done, info
