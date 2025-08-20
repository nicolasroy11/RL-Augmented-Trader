from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Dict, List
from RL.playground.stochastic.actor_critic import ActorCritic
from classes import Balances, OrderReport
from helpers import do_limit_buy, do_limit_sell, get_balances_snapshot, get_instant_notional_minimum
import runtime_settings
from services.core.dtos.full_single_long_cycle_dto import FullSingleLongCycleDto
from services.types import Actions

import pandas as pd
import numpy as np
import torch

from services.core.models import BTCFDUSDData
from services.rl_app.environments.base_environment import BaseTradingEnvironment
from services.data_app.repositories.data_repository import DataRepository


# ------------------------------
# Config
# ------------------------------
DATA_TABLE_NAME = BTCFDUSDData._meta.db_table
BASE_ASSET = runtime_settings.BASE_ASSET
QUOTE_ASSET = runtime_settings.QUOTE_ASSET
SYMBOL = f"{BASE_ASSET}{QUOTE_ASSET}"
DATA_FREQUENCY_SECS = runtime_settings.DATA_FREQUENCY_SECS
HISTORY_WINDOW_LENGTH = runtime_settings.DATA_TICKS_WINDOW
DECISION_MODE = "argmax"
POLICY_PATH = runtime_settings.RL_POLICY_PATH
client = runtime_settings.write_client

    
class TraderRepository():

    def __init__(self):
        self.current_cycle: FullSingleLongCycleDto = None
        self.run: List[FullSingleLongCycleDto] = []
        self.max_buy_quantity = 0.025
        self.data_repo = DataRepository()

    def run_single_buy_ppo_trader(self, num_cycles: int = 300) -> List[FullSingleLongCycleDto]:
        self.wait_for_sufficient_data()
        history_df = pd.DataFrame()
        self.load_inference_policy()

        while True:
            
            t0 = datetime.now()
            latest_window_data = self.data_repo.get_latest_tick_window_data(num_ticks=HISTORY_WINDOW_LENGTH)
            
            if len(latest_window_data) < HISTORY_WINDOW_LENGTH:
                time.sleep(DATA_FREQUENCY_SECS)
                continue

            data_df = BaseTradingEnvironment.to_env_dataframe(latest_window_data)
            env = BaseTradingEnvironment(data=data_df, window_size=HISTORY_WINDOW_LENGTH)
            env.current_step = len(latest_window_data)
            obs_vector = env.get_normalized_observation()

            last_tick = latest_window_data[-1]
            current_price = last_tick.price

            action_probs = self.compute_action_probs(obs_vector)
            trade_signal, qty = self.decide_trade(action_probs)
            print(f"Action: {trade_signal}")
            self.execute_trade(trade_signal, qty, current_price)

            history_df = pd.concat([history_df, pd.DataFrame([obs_vector])], ignore_index=True)
            if len(history_df) > HISTORY_WINDOW_LENGTH:
                history_df = history_df[-HISTORY_WINDOW_LENGTH:]

            if self.current_cycle is not None:
                unrealized_pnl = (current_price - self.current_cycle.entry_price) * self.current_cycle.position_size if self.current_cycle.position_size > 0 else 0.0

                print(
                    f"Current price: {current_price:.2f} | Entry: {self.current_cycle.entry_price:.2f} | Pos: {self.current_cycle.position_size:.6f} |  "
                    f"Unrealized: {unrealized_pnl:.2f} | Probs: { {k.name.lower(): v for k, v in action_probs.items()} } | Action: {trade_signal}"
                )

            if len(self.run):
                print(
                    f"Total profit so far: {sum([cycle.realized_pnl for cycle in self.run]):.2f}"
                )

            if len(self.run) >= num_cycles:
                return self.run

            t1 = datetime.now()
            time_to_sleep = DATA_FREQUENCY_SECS - (t1 - t0).total_seconds()
            if time_to_sleep < 0:
                time_to_sleep = 0

            time.sleep(time_to_sleep)

    
    def wait_for_sufficient_data(self):
        while True:
            latest_window_data = self.data_repo.get_latest_tick_window_data(num_ticks=HISTORY_WINDOW_LENGTH)
            
            if len(latest_window_data) < HISTORY_WINDOW_LENGTH:
                print(f"latest_window_data length: {len(latest_window_data)}")
                time.sleep(DATA_FREQUENCY_SECS)
                continue
            else: return


    def compute_action_probs(self, obs_vector) -> Dict[Actions, float]:
        with torch.no_grad():
            state_tensor = torch.tensor(obs_vector, dtype=torch.float32, device=self.policy_device).flatten().unsqueeze(0)
            logits, _ = self.policy(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            probs = dist.probs.squeeze(0).cpu().numpy()
            return {
                Actions.SELL: float(probs[0]),
                Actions.HOLD: float(probs[1]),
                Actions.BUY: float(probs[2]),
            }


    def decide_trade(self, action_probs: Dict[Actions, float]):
        if DECISION_MODE == "sample":
            labels = list(action_probs.keys())
            p = np.array(list(action_probs.values()), dtype=float)
            p = p / p.sum()
            trade_signal = np.random.choice(labels, p=p)
        else:
            trade_signal = max(action_probs, key=action_probs.get)

        qty = action_probs[Actions.BUY] * self.max_buy_quantity
        return trade_signal, qty
    

    def load_inference_policy(self):
        latest_window_data = self.data_repo.get_latest_tick_window_data(num_ticks=HISTORY_WINDOW_LENGTH)
        while len(latest_window_data) < HISTORY_WINDOW_LENGTH:
            print(f"latest_window_data length: {len(latest_window_data)}")
            time.sleep(DATA_FREQUENCY_SECS)
            latest_window_data = self.data_repo.get_latest_tick_window_data(num_ticks=HISTORY_WINDOW_LENGTH)

        dummy_df = BaseTradingEnvironment.to_env_dataframe(latest_window_data)
        dummy_env = BaseTradingEnvironment(data=dummy_df, window_size=HISTORY_WINDOW_LENGTH)
        dummy_env.current_step = len(dummy_df)
        dummy_obs_vector = dummy_env.get_normalized_observation()

        input_dim = len(dummy_obs_vector)
        action_dim = 3  # sell, hold, buy
        self.policy_device = torch.device("cpu")
        self.policy = ActorCritic(input_dim=input_dim, action_dim=action_dim).to(self.policy_device)

        state_dict = torch.load(POLICY_PATH, map_location=self.policy_device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
    

    def execute_trade(self, trade_signal: Actions, qty, price):
        notional = get_instant_notional_minimum(client=client, symbol=SYMBOL, price=price)
        if trade_signal == Actions.BUY and self.current_cycle is None:
            original_balances: Balances = get_balances_snapshot(client=client, base_asset=BASE_ASSET, quote_asset=QUOTE_ASSET)
            buy_order: OrderReport = do_limit_buy(client=client, symbol=SYMBOL, quantity=qty, price=price, id="none")
            qty = buy_order.filled_quantity
            if qty > notional:
                self.current_cycle = FullSingleLongCycleDto()
                self.current_cycle.start_time = datetime.now(tz=timezone.utc)
                self.current_cycle.position_size = qty
                self.current_cycle.entry_price = price
                self.current_cycle.initial_quote_balance = original_balances.free_quote_balance
        elif trade_signal == Actions.SELL and self.current_cycle is not None:
            original_balances: Balances = get_balances_snapshot(client=client, base_asset=BASE_ASSET, quote_asset=QUOTE_ASSET)
            sell_order: OrderReport = do_limit_sell(
                client=client,
                symbol=SYMBOL,
                quantity=original_balances.free_base_balance,
                high_limit=price,
                wait_for_completion=True,
                id="none"
            )
            new_balances: Balances = get_balances_snapshot(client=client, base_asset=BASE_ASSET, quote_asset=QUOTE_ASSET)
            self.current_cycle.exit_price = price
            self.current_cycle.exit_quote_balance = new_balances.free_quote_balance
            realized = self.current_cycle.exit_quote_balance - self.current_cycle.initial_quote_balance
            self.current_cycle.realized_pnl = realized
            self.current_cycle.end_time = datetime.now(tz=timezone.utc)
            self.run.append(self.current_cycle)
            self.current_cycle = None
