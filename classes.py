from typing import Any, Dict, List
from binance import Client
from datetime import datetime


class OrderSide:
    BUY = Client.SIDE_BUY
    SELL = Client.SIDE_SELL


class OrderType:
    MARKET = Client.ORDER_TYPE_MARKET
    LIMIT = Client.ORDER_TYPE_LIMIT


class OrderReport:
    id: str
    status: str
    side: OrderSide
    type: OrderType
    start_time: datetime
    end_time: datetime
    strike_price: float
    prefill_order: Dict
    postfill_order: Dict
    filled_quantity: float
    commission_fee: float
    time_it_took_to_fill_mins: float
    cummulative_quote_qty: float


class TradeCycleReport:
    def __init__(self):
        self.notes = ''
        self.start_time = None
        self.buy_time_indicators = {}

    id: str
    strategy_name: str
    start_time: datetime
    end_time: datetime
    start_balance: float
    end_balance: float
    buy_strike_price: float
    buy_quantity: float
    sell_strike_price: float
    sell_quantity: float
    orders: List[OrderReport]
    num_of_times_dca_used: int = 0
    time_it_took_to_fill_sell_mins: float
    is_test: bool
    investment_peak: float
    buy_time_indicators: Dict[str, Any] = {}
    notes: str


class TradeSeriesReport:
    def __init__(self):
        self.trade_cycles = []
        
    start_time: datetime
    end_time: datetime
    start_balance: float
    end_balance: float
    trade_cycles: List[TradeCycleReport]


class BidAskReport:
    ask_price: float
    ask_quantity: float
    bid_price: float
    bid_quantity: float
    spread: float


class SymbolParams:
    base_asset_precision: float
    min_notional: float
    min_lot_size: float


class Balances:
    free_base_balance: float
    locked_base_balance: float
    free_quote_balance: float
    locked_quote_balance: float
    
    def __init__(self, free_base_balance: float, locked_base_balance: float, free_quote_balance: float, locked_quote_balance: float):
        self.free_base_balance = free_base_balance
        self.locked_base_balance = locked_base_balance
        self.free_quote_balance = free_quote_balance
        self.locked_quote_balance = locked_quote_balance


class LoopReport:
    loop_number: str
    datetime: Any
    live_price: float
    open_pl: float
    free_quote_balance: float
    locked_quote_balance: float
    free_base_balance: float
    buy_price: float
    nominal_buy_amount: float
    filled_buy_amount: float
    sell_price: float
    num_open_orders: int
    ema_slope: float
    rsi: float
    
    def __init__(self):
        self.datetime = datetime.now()
        self.live_price = 0
        self.open_pl = 0
        self.free_quote_balance = 0
        self.locked_quote_balance = 0
        self.free_base_balance = 0
        self.buy_price = 0
        self.nominal_buy_amount = 0
        self.filled_buy_amount = 0
        self.sell_price = 0
        self.loop_number = 0
        self.num_open_orders = 0
        self.ema_slope = 0
        self.rsi = 0

