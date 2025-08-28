from datetime import datetime
from decimal import Decimal
from typing import Dict, List
from classes import Balances, OrderReport
from helpers import cancel_all_orders, do_limit_buy, do_limit_sell, do_market_sell, get_balances_snapshot, get_instant_notional_minimum
import runtime_settings
from services.core.dtos.full_single_long_cycle_dto import FullSingleLongCycleDto
from services.types import Actions
from binance.enums import SIDE_BUY, SIDE_SELL
import numpy as np

from services.core.models import FeatureSet, TickData, TickProbabilities, TradingSession, Transaction
from services.core.ML.configurations.PPO_flattened_history.environment import Environment
from services.data_app.repositories.data_repository import DataRepository


# ------------------------------
# Config
# ------------------------------
DATA_TABLE_NAME = TickData._meta.db_table
BASE_ASSET = runtime_settings.BASE_ASSET
QUOTE_ASSET = runtime_settings.QUOTE_ASSET
SYMBOL = f"{BASE_ASSET}{QUOTE_ASSET}"
DATA_FREQUENCY_SECS = runtime_settings.DATA_FREQUENCY_SECS
HISTORY_WINDOW_LENGTH = runtime_settings.DATA_TICKS_WINDOW_LENGTH
DECISION_MODE = "argmax"
client = runtime_settings.write_client

    
class TraderRepository():

    def __init__(self):
        self.current_buy: Transaction = None
        self.block_buys = False
        self.run: List[FullSingleLongCycleDto] = []
        self.max_buy_quantity = 0.025
        self.num_cycles: int = 300
        
        # if just a test session, reset balances
        if runtime_settings.IS_TESTNET:
            cancel_all_orders(client=client, symbol=SYMBOL)
            current_price = float(client.get_ticker(symbol=SYMBOL)['lastPrice'])
            notional = get_instant_notional_minimum(client=client, symbol=SYMBOL, price=current_price)
            original_balances: Balances = get_balances_snapshot(client=client, base_asset=BASE_ASSET, quote_asset=QUOTE_ASSET)
            if original_balances.free_base_balance >= notional:
                sell_order: OrderReport = do_market_sell(
                    client=client,
                    symbol=SYMBOL,
                    quantity=original_balances.free_base_balance
                )
            new_balances: Balances = get_balances_snapshot(client=client, base_asset=BASE_ASSET, quote_asset=QUOTE_ASSET)
            if new_balances.free_base_balance < notional: print('good to go')


    def run_single_buy_ppo_trader(self, feature_set_name: str) -> List[FullSingleLongCycleDto]:
        feature_set = FeatureSet.objects.filter(name=feature_set_name).first()
        if not feature_set:
            feature_set = FeatureSet.objects.filter(name='default').first()
        self.feature_set = feature_set
        trading_session = TradingSession()
        self.data_repo = DataRepository(feature_set_name=feature_set_name)
        trading_session.data_run = self.data_repo.data_run
        trading_session.feature_set = feature_set
        trading_session.save()
        self.trading_session = trading_session
        self.loop_number = 0
        self.start_process()


    def run_loop(self):
        self.loop_number += 1
        latest_window_data = self.data_repo.get_latest_tick_window_data_by_run(num_ticks=HISTORY_WINDOW_LENGTH, run_id=self.data_repo.data_run.id)
        
        if len(latest_window_data) < 5:
            print(f"latest_window_data: {len(latest_window_data)}")
            return
        
        env = Environment(tick_list=latest_window_data, feature_set=self.feature_set)
        action_probs = env.get_inference_action_probs()

        self.last_tick = latest_window_data[-1]
        current_price = self.last_tick.price

        trade_signal, qty = self.decide_trade(action_probs) 
        print(f"Action: {trade_signal}")

        self.execute_trade(trade_signal, qty, current_price)
        self.store_tick_probs(self.last_tick, action_probs)

        balances: Balances = get_balances_snapshot(client=client, base_asset=BASE_ASSET, quote_asset=QUOTE_ASSET)

        if balances.free_base_balance > 0:
            unrealized_pnl = (current_price - self.current_buy.strike_price) * self.current_buy.base_amount if self.current_buy.base_amount > 0 else 0.0

            print(
                f"Current price: {current_price:.2f} | Entry: {self.current_buy.strike_price:.2f} | Pos: {self.current_buy.base_amount:.6f} |  "
                f"Unrealized: {unrealized_pnl:.2f} | Probs: { {k.name.lower(): round(v, 4) for k, v in action_probs.items()} } | Action: {trade_signal}"
            )

        realized_pnl = self.get_realized_pnl()
        print(f'realized_pnl: {realized_pnl}')
        print(f'\nloop: {self.loop_number} ============================================================================================================================\n')


    def on_tick_stored(self, context):
        self.run_loop()


    def start_process(self):
        self.data_repo.start_data_collection(self.on_tick_stored)


    def store_tick_probs(self, last_tick: TickData, action_probs: Dict[Actions, float]):
        tick_probs = TickProbabilities()
        tick_probs.tick_data = last_tick
        tick_probs.buy_prob = action_probs[Actions.BUY]
        tick_probs.hold_prob = action_probs[Actions.HOLD]
        tick_probs.sell_prob = action_probs[Actions.SELL]
        tick_probs.save()


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
    

    def get_realized_pnl(self):
        pnl = Decimal("0.0")
        position = Decimal("0.0")
        entry_price = None
        for tx in list(Transaction.objects.filter(trading_session_id=self.trading_session.id).order_by("tick_data__timestamp")):
            price = Decimal(str(tx.strike_price))
            amount = Decimal(str(tx.base_amount))
            if tx.side == SIDE_BUY:
                if position == 0:
                    entry_price = price
                entry_price = ((entry_price * position) + (price * amount)) / (position + amount)
                position += amount
            elif tx.side == SIDE_SELL:
                if position > 0:
                    pnl += (price - entry_price) * min(position, amount)
                    position -= amount
                    if position == 0:
                        entry_price = None
                else:
                    raise NotImplementedError("Short trades not handled yet")
        return float(pnl)
    

    def execute_trade(self, trade_signal: Actions, qty, price):
        notional = get_instant_notional_minimum(client=client, symbol=SYMBOL, price=price)
        original_balances: Balances = get_balances_snapshot(client=client, base_asset=BASE_ASSET, quote_asset=QUOTE_ASSET)
        id = int(datetime.now().timestamp())
        if trade_signal == Actions.BUY and original_balances.free_base_balance <= notional and self.current_buy is None and not self.block_buys:
            self.block_buys = True
            buy_order: OrderReport = do_limit_buy(
                client=client,
                symbol=SYMBOL,
                quantity=qty,
                price=price,
                id=id
            )
            qty = buy_order.filled_quantity
            if qty > notional:
                transaction = Transaction()
                transaction.side = SIDE_BUY
                transaction.trading_session = self.trading_session
                transaction.strike_price = price
                transaction.base_amount = qty
                transaction.tick_data = self.last_tick
                transaction.save()
                self.current_buy = transaction
            else:
                self.block_buys = False

        elif trade_signal == Actions.SELL and original_balances.free_base_balance > notional:
            sell_order: OrderReport = do_limit_sell(
                client=client,
                symbol=SYMBOL,
                quantity=original_balances.free_base_balance,
                high_limit=price,
                wait_for_completion=True,
                id=id
            )
            new_balances: Balances = get_balances_snapshot(client=client, base_asset=BASE_ASSET, quote_asset=QUOTE_ASSET)
            transaction = Transaction()
            transaction.side = SIDE_SELL
            transaction.strike_price = price
            transaction.tick_data = self.last_tick
            transaction.base_amount = original_balances.free_base_balance - new_balances.free_base_balance
            transaction.trading_session = self.trading_session
            transaction.save()
            self.current_buy = None
            self.block_buys = False

