from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List
from django.apps import AppConfig
import time
import pandas as pd

from classes import Balances, OrderReport
from helpers import do_limit_buy, do_limit_sell, get_balances_snapshot, get_instant_notional_minimum
import runtime_settings
import numpy as np
import torch
from RL.playground.stochastic.actor_critic import ActorCritic
from services.types import Actions


class TradersConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "services.trader_app"

    def ready(self):
        client_test = runtime_settings.write_client.create_test_order(
            symbol='ETHUSDT',
            side='BUY',
            type='LIMIT',
            timeInForce='GTC',
            quantity=0.01,
            price='2000.00'
        )
        if client_test != {}:
            t = 0

        # from threading import Thread
        # self.current_cycle: FullCycle = None
        # self.run: List[FullCycle] = []
        # thread = Thread(target=self.run_bot, daemon=True)
        # thread.start()

    # def run_bot(self):
    #     from services.rl_app.environments.base_environment import BaseTradingEnvironment
    #     from services.data_app.repositories.data_repository import DataRepository
    #     data_repo = DataRepository()

    #     # ------------------------------
    #     # Config
    #     # ------------------------------
    #     BASE_ASSET = runtime_settings.BASE_ASSET
    #     QUOTE_ASSET = runtime_settings.QUOTE_ASSET
    #     SYMBOL = f"{BASE_ASSET}{QUOTE_ASSET}"
    #     DATA_FREQUENCY_SECS = runtime_settings.DATA_FREQUENCY_SECS
    #     HISTORY_WINDOW_LENGTH = runtime_settings.DATA_TICKS_WINDOW
    #     DECISION_MODE = "argmax"
    #     POLICY_PATH = runtime_settings.RL_POLICY_PATH
    #     MAX_BUY_QUANTITY = 0.025

    #     # ------------------------------
    #     # Binance client
    #     # ------------------------------

    #     client = runtime_settings.write_client

    #     # ------------------------------
    #     # Load RL policy using dummy env to determine input_dim
    #     # ------------------------------
    #     latest_window_data = data_repo.get_latest_tick_window_data(num_ticks=HISTORY_WINDOW_LENGTH)
    #     while len(latest_window_data) < HISTORY_WINDOW_LENGTH:
    #         time.sleep(DATA_FREQUENCY_SECS)
    #         latest_window_data = data_repo.get_latest_tick_window_data(num_ticks=HISTORY_WINDOW_LENGTH)

    #     dummy_df = BaseTradingEnvironment.to_env_dataframe(latest_window_data)
    #     dummy_env = BaseTradingEnvironment(data=dummy_df, window_size=HISTORY_WINDOW_LENGTH)
    #     dummy_env.current_step = len(dummy_df)
    #     dummy_obs_vector = dummy_env.get_normalized_observation()

    #     input_dim = len(dummy_obs_vector)
    #     action_dim = 3  # sell, hold, buy
    #     policy_device = torch.device("cpu")
    #     policy = ActorCritic(input_dim=input_dim, action_dim=action_dim).to(policy_device)

    #     state_dict = torch.load(POLICY_PATH, map_location=policy_device)
    #     policy.load_state_dict(state_dict)
    #     policy.eval()

    #     # ------------------------------
    #     # Action probabilities from normalized observation vector
    #     # ------------------------------
    #     def compute_action_probs(obs_vector):
    #         with torch.no_grad():
    #             state_tensor = torch.tensor(obs_vector, dtype=torch.float32, device=policy_device).flatten().unsqueeze(0)
    #             logits, _ = policy(state_tensor)
    #             dist = torch.distributions.Categorical(logits=logits)
    #             probs = dist.probs.squeeze(0).cpu().numpy()
    #             return {
    #                 Actions.SELL: float(probs[0]),
    #                 Actions.HOLD: float(probs[1]),
    #                 Actions.BUY: float(probs[2]),
    #             }

    #     def decide_trade(action_probs):
    #         if DECISION_MODE == "sample":
    #             labels = list(action_probs.keys())
    #             p = np.array(list(action_probs.values()), dtype=float)
    #             p = p / p.sum()
    #             trade_signal = np.random.choice(labels, p=p)
    #         else:
    #             trade_signal = max(action_probs, key=action_probs.get)

    #         qty = action_probs[Actions.BUY] * MAX_BUY_QUANTITY
    #         return trade_signal, qty

    #     def execute_trade(trade_signal: Actions, qty, price):
    #         notional = get_instant_notional_minimum(client=client, symbol=SYMBOL, price=price)
    #         if trade_signal == Actions.BUY and self.current_cycle is None:
    #             original_balances: Balances = get_balances_snapshot(client=client, base_asset=BASE_ASSET, quote_asset=QUOTE_ASSET)
    #             buy_order: OrderReport = do_limit_buy(client=client, symbol=SYMBOL, quantity=qty, price=price, id="none")
    #             qty = buy_order.filled_quantity
    #             if qty > notional:
    #                 self.current_cycle = FullCycle()
    #                 self.current_cycle.start_time = datetime.now(tz=timezone.utc)
    #                 self.current_cycle.position_size = qty
    #                 self.current_cycle.entry_price = price
    #                 self.current_cycle.initial_quote_balance = original_balances.free_quote_balance
    #         elif trade_signal == Actions.SELL and self.current_cycle is not None:
    #             original_balances: Balances = get_balances_snapshot(client=client, base_asset=BASE_ASSET, quote_asset=QUOTE_ASSET)
    #             sell_order: OrderReport = do_limit_sell(
    #                 client=client,
    #                 symbol=SYMBOL,
    #                 quantity=original_balances.free_base_balance,
    #                 high_limit=price,
    #                 wait_for_completion=True,
    #                 id="none"
    #             )
    #             new_balances: Balances = get_balances_snapshot(client=client, base_asset=BASE_ASSET, quote_asset=QUOTE_ASSET)
    #             self.current_cycle.exit_price = price
    #             self.current_cycle.exit_quote_balance = new_balances.free_quote_balance
    #             realized = self.current_cycle.exit_quote_balance - self.current_cycle.initial_quote_balance
    #             self.current_cycle.realized_pnl = realized
    #             self.current_cycle.end_time = datetime.now(tz=timezone.utc)
    #             self.run.append(self.current_cycle)
    #             self.current_cycle = None

    #     # ------------------------------
    #     # Main loop
    #     # ------------------------------
    #     history_df = pd.DataFrame()

    #     while True:
    #         t0 = datetime.now()
    #         latest_window_data = data_repo.get_latest_tick_window_data(num_ticks=HISTORY_WINDOW_LENGTH)
            
    #         if len(latest_window_data) < HISTORY_WINDOW_LENGTH:
    #             time.sleep(DATA_FREQUENCY_SECS)
    #             continue

    #         data_df = BaseTradingEnvironment.to_env_dataframe(latest_window_data)
    #         env = BaseTradingEnvironment(data=data_df, window_size=HISTORY_WINDOW_LENGTH)
    #         env.current_step = len(latest_window_data)
    #         obs_vector = env.get_normalized_observation()

    #         last_tick = latest_window_data[-1]
    #         current_price = last_tick.price

    #         action_probs = compute_action_probs(obs_vector)
    #         trade_signal, qty = decide_trade(action_probs)
    #         execute_trade(trade_signal, qty, current_price)

    #         history_df = pd.concat([history_df, pd.DataFrame([obs_vector])], ignore_index=True)
    #         if len(history_df) > HISTORY_WINDOW_LENGTH:
    #             history_df = history_df[-HISTORY_WINDOW_LENGTH:]

    #         if self.current_cycle is not None:
    #             unrealized_pnl = (current_price - self.current_cycle.entry_price) * self.current_cycle.position_size if self.current_cycle.position_size > 0 else 0.0

    #             print(
    #                 f"Current price: {current_price:.2f} | Entry: {self.current_cycle.entry_price:.2f} | Pos: {self.current_cycle.position_size:.6f} |  "
    #                 f"Unrealized: {unrealized_pnl:.2f} | Probs: { {k.name.lower(): v for k, v in action_probs.items()} } | Action: {trade_signal}"
    #             )

    #         if len(self.run):
    #             print(
    #                 f"Total profit so far: {sum([cycle.realized_pnl for cycle in self.run]):.2f}"
    #             )

    #         t1 = datetime.now()
    #         time_to_sleep = DATA_FREQUENCY_SECS - (t1 - t0).total_seconds()
    #         if time_to_sleep < 0:
    #             time_to_sleep = 0

    #         time.sleep(time_to_sleep)
