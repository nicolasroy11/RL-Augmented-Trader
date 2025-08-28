import numpy as np
import pandas as pd
from services.core.ML.configurations.PPO_flattened_history.environment import Environment
from services.core.models import FeatureSet

class StochasticSingleBuy(Environment):
    def __init__(self, feature_set: FeatureSet, tick_df: pd.DataFrame, initial_cash=1000000):
        super().__init__(tick_df=tick_df, feature_set=feature_set)
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holding = False
        self.buy_price = 0.0


    def step(self, action: int):
        """
        Actions:
            -1 = Sell
            0 = Hold
            1 = Buy

        Position state:
            self.position: 0 = flat, 1 = long
            self.entry_price: entry price if long, None if flat
        """
        price = self.data.iloc[self.current_step]['price']
        realized_pnl = 0.0

        # === Execute action ===
        if action == 1:  # BUY
            if self.position == 0:  # only open a new position if flat
                self.position = 1
                self.entry_price = price

        elif action == -1:  # SELL
            if self.position == 1:  # only sell if long
                realized_pnl = price - self.entry_price
                self.cash += realized_pnl  # accumulate realized gains/losses
                self.position = 0
                self.entry_price = None

        # HOLD (0) does nothing

        # === Advance environment ===
        self.current_step += 1
        done = self.current_step >= len(self.data)

        # === Get next observation ===
        obs = self.get_normalized_observation() if not done else None

        # === Reward: realized PnL only ===
        reward = realized_pnl

        # === Info dict for debugging/logging ===
        unrealized_pnl = (price - self.entry_price) if self.position == 1 else 0.0
        info = {
            "cash": self.cash,
            "position": self.position,
            "entry_price": self.entry_price if self.entry_price else 0.0,
            "price": price,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
        }

        return obs, reward, done, info


