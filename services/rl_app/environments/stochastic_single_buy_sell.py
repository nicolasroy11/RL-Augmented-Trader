import numpy as np
import pandas as pd
from services.rl_app.environments.base_environment import BaseTradingEnvironment

class StochasticSingleBuy(BaseTradingEnvironment):
    def __init__(self, data: pd.DataFrame, window_size=10, initial_cash=1000000):
        super().__init__(data, window_size)
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holding = False
        self.buy_price = 0.0

    def reset(self) -> np.ndarray:
        self.cash = self.initial_cash
        if self.initial_cash < self.data.iloc[self.window_size]['price']:
            raise ValueError("Initial cash too low to afford any asset.")
        self.holding = False
        self.buy_price = 0.0
        self.current_step = self.window_size
        return self._get_observation()

    def step(self, action: int):
        '''
        Actions: -1 = Sell, 0 = Hold, 1 = Buy
        '''
        price = self.data.iloc[self.current_step]['price']
        realized_pnl = 0.0

        if action == 1 and not self.holding:
            if self.cash >= price:
                self.holding = True
                self.buy_price = price
                self.cash -= price

        elif action == -1 and self.holding:
            self.holding = False
            self.cash += price
            realized_pnl = price - self.buy_price
            self.buy_price = 0.0

        self.current_step += 1
        done = self.current_step >= len(self.data)

        obs = self._get_normalized_observation() if not done else None
        reward = realized_pnl
        info = {
            'cash': self.cash,
            'holding': self.holding,
            'buy_price': self.buy_price,
            'price': price,
            'realized_pnl': realized_pnl
        }
        return obs, reward, done, info

