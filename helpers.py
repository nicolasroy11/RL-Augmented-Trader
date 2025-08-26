from ast import literal_eval
from datetime import datetime
import math
import time
from typing import Any, List
from classes import Balances, BidAskReport, LoopReport, OrderReport, OrderSide, OrderType, SymbolParams, TradeCycleReport
from binance import Client
import numpy as np
import pandas as pd
from binance import BinanceSocketManager
import os
import pandas_ta as ta


def get_raw_ohlcv_dataframe(client: Client, symbol: str, num_of_candles_back: int, candle_interval: str):
    klines = client.get_historical_klines(symbol=symbol, interval=candle_interval, limit=num_of_candles_back + 1)
    ohlcv_array = [np.float_(a[0:7]) for a in klines]
    temp_df = pd.DataFrame(ohlcv_array, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime'])
    temp_df['Datetime'] = [datetime.fromtimestamp(float(time)/1000) for time in temp_df['Time']]
    temp_df.set_index('Datetime', inplace=True)
    return temp_df


def get_latest_candles(client: Client, symbol: str, num_of_candles_back: int, candle_interval: str):
    klines = client.get_historical_klines(symbol=symbol, interval=candle_interval, limit=num_of_candles_back + 1)
    ohlcv_array = [np.float_(a[0:7]) for a in klines]
    temp_df = pd.DataFrame(ohlcv_array, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime'])
    temp_df['Datetime'] = [datetime.fromtimestamp(float(time)/1000) for time in temp_df['Time']]
    temp_df.set_index('Datetime', inplace=True)
    return temp_df


def get_latest_closed_candles(client: Client, symbol: str, candle_interval: str, num_of_candles_back: int=None, start_datetime: datetime=None, end_datetime: datetime=None) -> pd.DataFrame:
    if start_datetime and end_datetime:
        klines = client.get_historical_klines(symbol=symbol, interval=candle_interval, start_str=int(start_datetime.timestamp())*1000, end_str=int(end_datetime.timestamp())*1000)
    else:
        klines = client.get_historical_klines(symbol=symbol, interval=candle_interval, limit=num_of_candles_back)
    ohlcv_array = [np.float_(a[0:7]) for a in klines]
    temp_df = pd.DataFrame(ohlcv_array, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime'])
    temp_df['Datetime'] = [datetime.fromtimestamp(float(time)/1000) for time in temp_df['Time']]
    temp_df.set_index('Datetime', inplace=True)
    temp_df['CloseTime'] = [datetime.fromtimestamp(float(time)/1000) for time in temp_df['CloseTime']]
    temp_df['IsOpen'] = temp_df['CloseTime'] > datetime.now()
    temp_df.drop(temp_df.loc[temp_df['IsOpen']==True].index, inplace=True)
    temp_df.drop(columns=['CloseTime', 'IsOpen'], inplace=True)
    return temp_df


def get_last_low(client: Client, symbol: str, candle_interval: str):
    last_close = get_latest_closed_candles(client=client, symbol=symbol, candle_interval=candle_interval, num_of_candles_back=1)
    return float(last_close['Low'][-1])


def get_last_close(client: Client, symbol: str, candle_interval: str):
    last_close = get_latest_closed_candles(client=client, symbol=symbol, candle_interval=candle_interval, num_of_candles_back=1)
    return float(last_close['Close'][-1])


def get_heikin_ashi_df_from_data_df(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df
    heikin_ashi_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close'])
    heikin_ashi_df['Close'] = (temp_df['Open'] + temp_df['High'] + temp_df['Low'] + temp_df['Close']) / 4
    for i in range(len(temp_df)):
        if i == 0:
            heikin_ashi_df.iat[0, 0] = temp_df['Open'].iloc[0]
        else:
            heikin_ashi_df.iat[i, 0] = (heikin_ashi_df.iat[i-1, 0] + heikin_ashi_df.iat[i-1, 3]) / 2
    heikin_ashi_df['High'] = heikin_ashi_df.loc[:, ['Open', 'Close']].join(temp_df['High']).max(axis=1)
    heikin_ashi_df['Low'] = heikin_ashi_df.loc[:, ['Open', 'Close']].join(temp_df['Low']).min(axis=1)
    heikin_ashi_df['Datetime'] = [datetime.fromtimestamp(float(time)/1000) for time in temp_df['Time']]
    heikin_ashi_df.set_index('Datetime', inplace=True)
    return heikin_ashi_df


def get_heikin_ashi_df(client: Client, symbol: str, num_of_candles_back: int, candle_interval: str, closed_candles_only=True) -> pd.DataFrame:
    if closed_candles_only:
        temp_df = get_latest_closed_candles(client=client, symbol=symbol, num_of_candles_back=num_of_candles_back, candle_interval=candle_interval)
    else:
        temp_df = get_raw_ohlcv_dataframe(client, symbol, num_of_candles_back, candle_interval)
    heikin_ashi_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close'])
    heikin_ashi_df['Close'] = (temp_df['Open'] + temp_df['High'] + temp_df['Low'] + temp_df['Close']) / 4
    for i in range(len(temp_df)):
        if i == 0:
            heikin_ashi_df.iat[0, 0] = temp_df['Open'].iloc[0]
        else:
            heikin_ashi_df.iat[i, 0] = (heikin_ashi_df.iat[i-1, 0] + heikin_ashi_df.iat[i-1, 3]) / 2
    heikin_ashi_df['High'] = heikin_ashi_df.loc[:, ['Open', 'Close']].join(temp_df['High']).max(axis=1)
    heikin_ashi_df['Low'] = heikin_ashi_df.loc[:, ['Open', 'Close']].join(temp_df['Low']).min(axis=1)
    heikin_ashi_df['Datetime'] = [datetime.fromtimestamp(float(time)/1000) for time in temp_df['Time']]
    heikin_ashi_df.set_index('Datetime', inplace=True)
    return heikin_ashi_df


def get_bid_ask_report(client: Client, symbol: str) -> BidAskReport:
    report = BidAskReport()
    ticker = client.get_ticker(symbol=symbol)
    report.ask_price = float(ticker['askPrice'])
    report.ask_quantity = float(ticker['askQty'])
    report.bid_price = float(ticker['bidPrice'])
    report.bid_quantity = float(ticker['bidQty'])
    report.spread = round((1 - float(ticker['bidPrice']) / float(ticker['askPrice'])) * 100, 5)
    return report


def get_buytime_spread(client: Client, symbol: str) -> float:
    ticker = client.get_ticker(symbol=symbol)
    return round((1 - float(ticker['askPrice']) / float(ticker['lastPrice'])) * 100, 5)


def get_selltime_spread(client: Client, symbol: str) -> float:
    ticker = client.get_ticker(symbol=symbol)
    return round((1 - float(ticker['bidPrice']) / float(ticker['lastPrice'])) * 100, 5)


def do_limit_buy(client: Client, symbol: str, price: float, quantity: float, id: str, print_logs=True) -> OrderReport:
    
    # check notional minimum sell amount
    symbol_filter_dict = {filter['filterType']: filter for filter in client.get_symbol_info(symbol=symbol)['filters']}
    if 'MIN_NOTIONAL' in symbol_filter_dict:
            notional_key = 'MIN_NOTIONAL' 
    else: 
        notional_key = 'NOTIONAL'
    notional_minimum = float(symbol_filter_dict[notional_key]['minNotional'])/price

    print(f'    placing limit buy for {round(quantity, 5)} {symbol} at {price} - {str(datetime.now())}...')
    report = OrderReport()
    report.type = OrderType.LIMIT
    report.side = OrderSide.BUY
    report.start_time = datetime.now()
    report.strike_price = price
    
    print(f'    strike_price: {report.strike_price}')
    buy_order = client.order_limit_buy(
        symbol=f'{symbol}',
        newClientOrderId=id,
        quantity=str(np.format_float_positional(round(quantity, 5), trim='-')),
        price=str(round(price, 2)),
    )

    print(f'    awaiting buy fulfillment...')
    buy_order_id = buy_order['orderId']
    report.prefill_order = buy_order

    buy_order_filled = False
    buy_order_cancelled = False
    max_fill_attempt_duration_seconds = 15
    counter_start = datetime.now()

    # loop until the whole order is done
    last_status = buy_order["status"]
    while not buy_order_filled and not buy_order_cancelled:
        time.sleep(0.3)
        # check for status until all filled
        buy_order = client.get_order(symbol=symbol, orderId=buy_order_id)

        if buy_order["status"] != last_status:
            print(f'            status: {buy_order["status"]}')
            last_status = buy_order["status"]

        if buy_order["status"] == client.ORDER_STATUS_CANCELED:
            buy_order_cancelled = True
            continue

        if buy_order['status'] == client.ORDER_STATUS_PARTIALLY_FILLED:
            executed_qty = float(buy_order["executedQty"])
            if executed_qty > notional_minimum:
                print(f'            partial fill sufficient at {executed_qty} > {notional_minimum}')
            else:
                print(f'            partial fill insufficient at {executed_qty} < {notional_minimum}')
                counter_end = datetime.now()
                duration = counter_end - counter_start
                duration_in_s = duration.total_seconds()
                if duration_in_s > max_fill_attempt_duration_seconds:
                    try:
                        client.cancel_order(
                            symbol=symbol,
                            orderId=buy_order_id
                        )
                        cancel_order_done = False
                        while not cancel_order_done:
                            # check for status until canceled
                            buy_order = client.get_order(symbol=symbol, orderId=buy_order_id)
                            report.filled_quantity = float(buy_order['executedQty'])    # updating filled quantity just in case it changes during cancelation
                            print(f'    cancel_order_status: {buy_order["status"]}')
                            if buy_order['status'] == client.ORDER_STATUS_CANCELED:
                                cancel_order_done = True
                                print(f'    order canceled\n')
                                break
                        buy_order_cancelled = True
                    except:
                        pass
                else:
                    continue

        if buy_order['status'] == client.ORDER_STATUS_FILLED:
            report.postfill_order = buy_order
            break

        # if it takes too long to fill the buy, we bail
        else:
            counter_end = datetime.now()
            duration = counter_end - counter_start
            duration_in_s = duration.total_seconds()
            if duration_in_s > max_fill_attempt_duration_seconds:
                try:
                    client.cancel_order(
                        symbol=symbol,
                        orderId=buy_order_id
                    )
                    cancel_order_done = False
                    while not cancel_order_done:
                        # check for status until canceled
                        buy_order = client.get_order(symbol=symbol, orderId=buy_order_id)
                        report.filled_quantity = float(buy_order['executedQty'])    # updating filled quantity just in case it changes during cancelation
                        print(f'    cancel_order_status: {buy_order["status"]}')
                        if buy_order['status'] == client.ORDER_STATUS_CANCELED:
                            cancel_order_done = True
                            print(f'    order canceled\n')
                            break
                    buy_order_cancelled = True
                except:
                    pass

    try:
        buy_order = client.get_order(symbol=symbol, orderId=buy_order_id)   # we get the latest copy just in case; try-catch used in case order was canceled and is 404
    except:
        pass    # otherwise we just use the last copy we have in memory
    report.filled_quantity = float(buy_order['executedQty'])
    if report.filled_quantity > 0:
        print(f'    total buy: {buy_order["cummulativeQuoteQty"]} {symbol}')
        report.cummulative_quote_qty = float(buy_order["cummulativeQuoteQty"])
    # report.commission_fee = buy_order['commission']
    report.status = buy_order["status"]
    report.end_time = datetime.now()
    return report


def do_futures_limit_buy(client: Client, symbol: str, price: float, quantity: float, id: str, print_logs=True) -> OrderReport:
    
    # check notional minimum sell amount
    symbol_filter_dict = {filter['filterType']: filter for filter in client.get_symbol_info(symbol=symbol)['filters']}
    if 'MIN_NOTIONAL' in symbol_filter_dict:
            notional_key = 'MIN_NOTIONAL' 
    else: 
        notional_key = 'NOTIONAL'
    notional_minimum = float(symbol_filter_dict[notional_key]['minNotional'])/price

    print(f'    placing limit buy for {round(quantity, 5)} {symbol} at {price} - {str(datetime.now())}...')
    report = OrderReport()
    report.type = OrderType.LIMIT
    report.side = OrderSide.BUY
    report.start_time = datetime.now()
    report.strike_price = price
    
    print(f'    strike_price: {report.strike_price}')
    buy_order = client.futures_create_order(
        symbol=symbol,
        # newClientOrderId=id,
        type=OrderType.LIMIT,
        timeInForce=client.TIME_IN_FORCE_GTC,  # Can be changed - see link to API doc below
        price=round(price),  # The price at which you wish to buy/sell, float
        side=Client.SIDE_BUY,  # Direction ('BUY' / 'SELL'), string
        quantity=round(quantity, 3)  # Number of coins you wish to buy / sell, float
    )

    print(f'    awaiting buy fulfillment...')
    buy_order_id = buy_order['orderId']
    report.id = buy_order_id
    report.prefill_order = buy_order

    buy_order_filled = False
    buy_order_cancelled = False
    max_fill_attempt_duration_seconds = 30
    counter_start = datetime.now()

    # loop until the whole order is done
    last_status = buy_order["status"]
    while not buy_order_filled and not buy_order_cancelled:
        time.sleep(0.3)
        # check for status until all filled
        buy_order = client.futures_get_order(symbol=symbol, orderId=buy_order_id)

        if buy_order["status"] != last_status:
            print(f'            status: {buy_order["status"]}')
            last_status = buy_order["status"]

        if buy_order["status"] == client.ORDER_STATUS_CANCELED:
            buy_order_cancelled = True
            continue

        if buy_order['status'] == client.ORDER_STATUS_PARTIALLY_FILLED:
            executed_qty = float(buy_order["executedQty"])
            if executed_qty > notional_minimum:
                print(f'            partial fill sufficient at {executed_qty} > {notional_minimum}')
            else:
                print(f'            partial fill insufficient at {executed_qty} < {notional_minimum}')
                counter_end = datetime.now()
                duration = counter_end - counter_start
                duration_in_s = duration.total_seconds()
                if duration_in_s > max_fill_attempt_duration_seconds:
                    try:
                        client.futures_cancel_order(
                            symbol=symbol,
                            orderId=buy_order_id
                        )
                        cancel_order_done = False
                        while not cancel_order_done:
                            # check for status until canceled
                            buy_order = client.futures_get_order(symbol=symbol, orderId=buy_order_id)
                            report.filled_quantity = float(buy_order['executedQty'])    # updating filled quantity just in case it changes during cancelation
                            print(f'    cancel_order_status: {buy_order["status"]}')
                            if buy_order['status'] == client.ORDER_STATUS_CANCELED:
                                cancel_order_done = True
                                print(f'    order canceled\n')
                                break
                        buy_order_cancelled = True
                    except:
                        pass
                else:
                    continue

        if buy_order['status'] == client.ORDER_STATUS_FILLED:
            report.postfill_order = buy_order
            break

        # if it takes too long to fill the buy, we bail
        else:
            counter_end = datetime.now()
            duration = counter_end - counter_start
            duration_in_s = duration.total_seconds()
            if duration_in_s > max_fill_attempt_duration_seconds:
                try:
                    client.futures_cancel_order(
                        symbol=symbol,
                        orderId=buy_order_id
                    )
                    cancel_order_done = False
                    while not cancel_order_done:
                        # check for status until canceled
                        buy_order = client.futures_get_order(symbol=symbol, orderId=buy_order_id)
                        report.filled_quantity = float(buy_order['executedQty'])    # updating filled quantity just in case it changes during cancelation
                        print(f'    cancel_order_status: {buy_order["status"]}')
                        if buy_order['status'] == client.ORDER_STATUS_CANCELED:
                            cancel_order_done = True
                            print(f'    order canceled\n')
                            break
                    buy_order_cancelled = True
                except:
                    pass
    try:
        buy_order = client.futures_get_order(symbol=symbol, orderId=buy_order_id)   # we get the latest copy just in case; try-catch used in case order was canceled and is 404
    except:
        pass    # otherwise we just use the last copy we have in memory
    report.filled_quantity = float(buy_order['executedQty'])
    if report.filled_quantity > 0:
        print(f'    total buy: {buy_order["cumQuote"]} {symbol}')
        report.cummulative_quote_qty = float(buy_order["cumQuote"])
    # report.commission_fee = buy_order['commission']
    report.status = buy_order["status"]
    report.end_time = datetime.now()
    return report


def do_lightning_limit_buy(client: Client, symbol: str, price: float, quantity: float, id: str, print_logs=True) -> OrderReport:
    
    # check ticker for bid/ask spread
    print(f'    placing limit buy for {quantity} {symbol} at {price} - {str(datetime.now())}...')
    report = OrderReport()
    report.type = OrderType.LIMIT
    report.side = OrderSide.BUY
    report.start_time = datetime.now()
    report.strike_price = price
    
    print(f'    strike_price: {report.strike_price}')
    buy_order = client.order_limit_buy(
        symbol=f'{symbol}',
        newClientOrderId=id,
        quantity=str(np.format_float_positional(round(quantity, 5), trim='-')),
        price=str(round(price, 2)),
    )

    print(f'    awaiting buy fulfillment...')
    buy_order_id = buy_order['orderId']
    report.prefill_order = buy_order

    buy_order_filled = False
    buy_order_cancelled = False
    counter_start = datetime.now()

    # loop until the whole order is done
    partial_fill_cancel_countdown = 1
    last_status = buy_order["status"]
    while not buy_order_filled and not buy_order_cancelled:
        time.sleep(0.3)
        # check for status until all filled
        buy_order = client.get_order(symbol=symbol, orderId=buy_order_id)

        if buy_order["status"] != last_status:
            print(f'            status: {buy_order["status"]}')
            last_status = buy_order["status"]

        if buy_order['status'] == client.ORDER_STATUS_PARTIALLY_FILLED:
            partial_fill_cancel_countdown -= 1

        if buy_order['status'] == client.ORDER_STATUS_FILLED:
            # print(f'    post-fill buy order: {buy_order}')
            print(f'    total buy: {buy_order["cummulativeQuoteQty"]} {symbol}')
            report.postfill_order = buy_order
            buy_order_filled = True

        # if it takes too long to fill the buy, we bail
        else:
            counter_end = datetime.now()
            duration = counter_end - counter_start
            duration_in_s = duration.total_seconds()
            if duration_in_s > 2:
                client.cancel_order(
                    symbol=symbol,
                    orderId=buy_order_id
                )
                cancel_order_done = False
                while not cancel_order_done:
                    # check for status until all filled
                    buy_order = client.get_order(symbol=symbol, orderId=buy_order_id)
                    print(f'    cancel_order_status: {buy_order["status"]}')
                    if buy_order['status'] == client.ORDER_STATUS_CANCELED:
                        cancel_order_done = True
                        print(f'    order canceled\n')
                        break
                buy_order_cancelled = True

    report.filled_quantity = float(buy_order['executedQty'])
    report.status = buy_order["status"]
    report.end_time = datetime.now()
    return report


def do_market_buy(client: Client, symbol: str, quantity: float, print_logs=True) -> OrderReport:
    
    # TODO: check ticker for bid/ask spread
    
    report = OrderReport()
    report.type = OrderType.MARKET
    report.side = OrderSide.BUY
    report.start_time = datetime.now()

    buy_order_filled = False
    buy_order_expired = False
    buy_order = client.order_market_buy(
        symbol=symbol,
        quantity=round(quantity, 5)
    )
    report.strike_price = float(buy_order['fills'][0]['price'])
    print(f'    strike_price: {report.strike_price}')
    report.prefill_order = buy_order

    buy_order_id = buy_order['orderId']
    # print(f'    pre-fill buy_order: {buy_order}')

    # check for status until all filled
    while not (buy_order_filled or buy_order_expired):
        time.sleep(2)
        buy_order = client.get_order(symbol=symbol, orderId=buy_order_id)
        print(f'    buy status: {buy_order["status"]}')
        if buy_order['status'] == client.ORDER_STATUS_EXPIRED: buy_order_expired = True

        if buy_order['status'] == client.ORDER_STATUS_FILLED:
            # print(f'    post-fill buy order: {buy_order}')
            report.postfill_order = buy_order

            buy_order_filled = True
        else:
            continue
    
    report.filled_quantity = float(buy_order['executedQty'])
    # report.commission_fee = buy_order['commission']
    report.status = buy_order["status"]
    report.end_time = datetime.now()
    return report


def do_market_sell(client: Client, symbol: str, quantity: float, print_logs=True) -> OrderReport:
    report = OrderReport()
    report.type = OrderType.MARKET
    report.side = OrderSide.SELL
    report.start_time = datetime.now()

    sell_order_filled = False
    sell_order_expired = False
    sell_order = client.order_market_sell(
        symbol=symbol,
        quantity=round(quantity, 5)
    )
    report.prefill_order = sell_order

    sell_order_id = sell_order['orderId']
    report.id = sell_order_id

    # check for status until all filled
    while not (sell_order_filled or sell_order_expired):
        time.sleep(2)
        sell_order = client.get_order(symbol=symbol, orderId=sell_order_id)
        print(f'            sell status: {sell_order["status"]}')
        if sell_order['status'] == client.ORDER_STATUS_EXPIRED: sell_order_expired = True

        if sell_order['status'] == client.ORDER_STATUS_FILLED:
            # print(f'            post-fill sell order: {sell_order}')
            report.postfill_order = sell_order

            sell_order_filled = True
        else:
            continue
    
    # here, you should return an object with how much was sold in the end and the strike price
    report.filled_quantity = sell_order['executedQty']
    report.status = sell_order["status"]
    report.end_time = datetime.now()
    return report


def do_limit_sell(client: Client, symbol: str, quantity: float, high_limit: float, id: str, wait_for_completion = True) -> OrderReport:
    print(f'            placing Limit sell with limit at {round(high_limit * (1 - 0.000001), 5)}')

    report = OrderReport()
    report.type = OrderType.LIMIT
    report.side = OrderSide.SELL
    report.start_time = datetime.now()
    report.strike_price = high_limit
    print(f'            strike_price: {report.strike_price}')
    
    # TODO: explore iceberg sells to see how much faster it goes
    t0 = datetime.now()
    sell_order = client.order_limit_sell(
        symbol=symbol,
        newClientOrderId=id,
        quantity=np.format_float_positional(round(quantity, 5), trim='-'),
        price=str(round(high_limit * (1 - 0.000001), 2))
    )
    sell_order_id = sell_order['orderId']
    report.id = sell_order_id
    report.prefill_order = sell_order

    # check for status until all filled
    sell_order_filled = False
    last_status = sell_order["status"]

    if wait_for_completion:
        while not sell_order_filled:
            time.sleep(0.3)
            if sell_order["status"] != last_status:
                print(f'            status: {sell_order["status"]}')
            last_status = sell_order["status"]
            sell_order = client.get_order(symbol=symbol, orderId=sell_order_id)
            if sell_order['status'] == client.ORDER_STATUS_PARTIALLY_FILLED: continue
            if sell_order['status'] == Client.ORDER_STATUS_EXPIRED:
                print(f'Sell Order expired!; trying again')
                do_limit_sell(client, symbol, quantity, high_limit)
            if sell_order['status'] == client.ORDER_STATUS_FILLED:
                print(f'            status: {sell_order["status"]}')
                report.postfill_order = sell_order
                t1 = datetime.now()
                report.time_it_took_to_fill_mins = round((t1-t0).seconds/60, 2)
                print(f'sell completed in {report.time_it_took_to_fill_mins} mins')
                sell_order_filled = True
            # TODO: here check for partially filled orders taking too long to fill and sell at market
            # TODO: cancel NEW order taking too long and rebuy
    
            
    # here, you should return an object with how much was sold in the end and the strike price
    report.filled_quantity = float(sell_order['executedQty'])
    report.status = sell_order["status"]
    report.end_time = datetime.now()
    return report


def cancel_order(client: Client, symbol: str, order_id: str):
    client.cancel_order(
        symbol=symbol,
        orderId=order_id
    )
    while True:
        # check for status until all filled
        order = client.get_order(symbol=symbol, orderId=order_id)
        print(f'    cancel_order_status: {order["status"]}')
        if order['status'] == client.ORDER_STATUS_CANCELED:
            print(f'    order canceled\n')
            break
    return True


def cancel_all_orders(client: Client, symbol: str):
    orders = client.get_open_orders(symbol=symbol)
    for order in orders:
        id = order['orderId']
        client.cancel_order(
            symbol=symbol,
            orderId=id
        )
        while True:
            # check for status until all filled
            order = client.get_order(symbol=symbol, orderId=id)
            print(f'    cancel_order_status: {order["status"]}')
            if order['status'] == client.ORDER_STATUS_CANCELED:
                print(f'    order canceled\n')
                break
        return True


def futures_cancel_order(client: Client, symbol: str, order_id: str):
    client.futures_cancel_order(
        symbol=symbol,
        orderId=order_id
    )
    while True:
        # check for status until all filled
        order = client.futures_get_order(symbol=symbol, orderId=order_id)
        print(f'    cancel_order_status: {order["status"]}')
        if order['status'] == client.ORDER_STATUS_CANCELED:
            print(f'    order canceled\n')
            break
    return True


def do_timed_limit_sell(client: Client, symbol: str, quantity: float, high_limit: float, id: str, time_limit_secs: int) -> OrderReport:
    print(f'            placing Limit sell with limit at {round(high_limit * (1 - 0.00001), 5)}')

    report = OrderReport()
    report.type = OrderType.LIMIT
    report.side = OrderSide.SELL
    report.start_time = datetime.now()
    report.strike_price = high_limit
    print(f'            strike_price: {report.strike_price}')
    
    t0 = datetime.now()
    sell_order = client.order_limit_sell(
        symbol=symbol,
        quantity=str(round(quantity, 5)),
        price=str(round(high_limit * (1 - 0.00001), 2))
    )
    sell_order_id = sell_order['orderId']
    report.prefill_order = sell_order

    # check for status until all filled
    sell_order_filled = False
    last_status = sell_order["status"]
    while not sell_order_filled:
        time.sleep(time_limit_secs)
        if sell_order["status"] != last_status:
            print(f'            status: {sell_order["status"]}')
        last_status = sell_order["status"]
        sell_order = client.get_order(symbol=symbol, orderId=sell_order_id)

        # if sell_order['status'] == client.ORDER_STATUS_NEW:
        #     continue
        if sell_order['status'] == client.ORDER_STATUS_PARTIALLY_FILLED or sell_order['status'] == client.ORDER_STATUS_NEW:
            live_price = float(client.get_ticker(symbol=f'{symbol}')['lastPrice'])
            print(f'trying sell again at {live_price}')
            unsold_amount_so_far = float(sell_order['origQty']) - float(sell_order['executedQty'])
            print(f'unsold_amount_so_far {unsold_amount_so_far}')
            cancel_order(client=client, symbol=symbol, order_id=sell_order['orderId'])
            sell_order = client.order_limit_sell(
                symbol=symbol,
                quantity=str(round(unsold_amount_so_far, 5)),
                price=str(round(live_price, 2))
            )
            sell_order_id = sell_order['orderId']
            report.prefill_order = sell_order
        if sell_order['status'] == client.ORDER_STATUS_FILLED:
            print(f'            status: {sell_order["status"]}')
            report.postfill_order = sell_order
            t1 = datetime.now()
            report.time_it_took_to_fill_mins = round((t1-t0).seconds/60, 2)
            print(f'sell completed in {report.time_it_took_to_fill_mins} mins')
            sell_order_filled = True

            
    # here, you should return an object with how much was sold in the end and the strike price
    report.filled_quantity = float(sell_order['executedQty'])
    # report.commission_fee = sell_order['commission']
    report.status = sell_order["status"]
    report.end_time = datetime.now()
    return report


async def do_trailing_stop_loss_market_sell(client: Client, symbol: str, quantity: float, starting_benchmark: float, TSL_tolerance: float) -> OrderReport:
    bsm = BinanceSocketManager(client)
    socket = bsm.trade_socket(symbol)

    live_prices = []
    highest_high_since_buy = starting_benchmark
    TSL = starting_benchmark * TSL_tolerance
    
    sell_order_done = False
    while not sell_order_done:
        # get live price from socket
        try:
            await socket.__aenter__()
            msg = await socket.recv()
            live_price = float(msg['p'])
            live_prices.append(live_price)
            if len(live_prices) < 2: continue
            print(f'            sell mode current price: {live_price} ({round((1 - live_price/starting_benchmark) * 100, 2)}%)')
        except BaseException as e:
            print(f'            failed to get price from socket: {e.args[0]}')
            continue

        if live_price > highest_high_since_buy:
            highest_high_since_buy = live_price
            TSL = highest_high_since_buy * (1 - 0.0002)

        if live_price/live_prices[-2] < 1: # if the price either starts to stagnate or go negative at all, we bump up the TSL faster
            # tls_level *= 1.00002
            # TSL *= tls_level
            TSL = TSL * (1 + 0.00001)
        # else: 
        #     tls_level = TSL_tolerance

        # Strategy 2: any price action makes the 
        # diff = live_price - live_prices[-2] 
        # if diff <= 0: # if the price either starts to stagnate or go negative at all, we bump up the TSL faster
        #     TSL = TSL + abs(diff)
        # else:
        #     TSL = TSL + diff/2
        
        

        print('                ------------------------------------------------')
        print(f'                current highest_high_since_buy: {str(highest_high_since_buy)}')
        print(f'                current trailing stop loss: {str(TSL)}')
        print('                ------------------------------------------------')
        
        if live_price < TSL:
            print( f'            placing Market sell with limit at live_price {str("%.2f" % live_price)}')
            # sell_order = do_limit_sell(client=client, symbol=symbol, quantity=quantity, high_limit=live_price)
            sell_order = do_market_sell(client=client, symbol=symbol, quantity=quantity, print_logs=True)
            sell_order_done = True
        else:
            continue
    return sell_order


import csv
def log_trade(start_time: datetime, end_time: datetime, trade_duration: float, start_balance: float, end_balance: float, trade_profit: float, trade_profit_pct: float, buy_strike_price: float, buy_quantity: float, sell_strike_price: str, sell_quantity: float, num_of_times_dca_used: int):
    with open('traderplus/trades.csv', mode='a') as trades_file:
        trades_csv = csv.writer(trades_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        trades_csv.writerow([start_time, end_time, trade_duration, buy_strike_price, sell_strike_price, buy_quantity, sell_quantity, num_of_times_dca_used, start_balance, end_balance, f'{trade_profit_pct}%', f'${trade_profit}'])


def log_strategy_params(params_string: str, file_path: str = 'traderplus/trades.csv'):
    with open(file_path, mode='a') as trades_file:
        trades_csv = csv.writer(trades_file, delimiter=',')
        trades_csv.writerow([params_string])


def log_message_csv(message: str, file_path: str = 'traderplus/trades.csv'):
    with open(file_path, mode='a') as trades_file:
        trades_csv = csv.writer(trades_file, delimiter=',')
        trades_csv.writerow([message])


def log_loop_csv(loop_report: LoopReport, file_path: str):
    with open(file_path, mode='a') as trades_file:
        trades_csv = csv.writer(trades_file, delimiter=',')
        trades_csv.writerow([
            loop_report.loop_number,
            loop_report.datetime,
            loop_report.live_price,
            loop_report.open_pl,
            loop_report.free_quote_balance,
            loop_report.locked_quote_balance,
            loop_report.free_base_balance,
            loop_report.buy_price,
            loop_report.nominal_buy_amount,
            loop_report.filled_buy_amount,
            loop_report.sell_price,
            loop_report.num_open_orders,
            loop_report.rsi,
            loop_report.ema_slope
        ])


def log_message(message: str, file_path: str):
    with open(file_path, mode='a') as file:
        file.writelines(message)


def log_strategy_trade(report: TradeCycleReport, file_path: str = 'traderplus/trades.csv'):
    trade_profit = round(report.end_balance - report.start_balance, 2)
    investment_profit_pct = round((trade_profit/report.investment_peak), 4) * 100
    trade_percent_return = (report.end_balance/report.start_balance) * 100
    trade_profit_log_pct = np.log(report.end_balance/report.start_balance) * 100  # R=ln(Vf/Vi)t×100
    with open(file_path, mode='a') as trades_file:
        trades_csv = csv.writer(trades_file, delimiter=',')
        trades_csv.writerow([
            report.id,
            # report.strategy_name,
            report.start_time.strftime("%m-%d-%Y, %H:%M:%S"),
            report.end_time.strftime("%m-%d-%Y, %H:%M:%S"),
            f'{round((report.end_time - report.start_time).total_seconds()/60, 3)} minutes',
            round(report.buy_strike_price, 2),
            round(report.sell_strike_price, 2),
            round(report.buy_quantity, 2),
            round(report.sell_quantity, 5),
            report.num_of_times_dca_used,
            round(report.start_balance, 2),
            round(report.end_balance, 2),
            f'{investment_profit_pct}%',
            f'{round(trade_profit_log_pct, 7)}%',
            f'${trade_profit}',
            report.buy_time_indicators,
            report.notes])


def log_straight_shot_strategy_trade(report: TradeCycleReport, file_path: str = 'traderplus/trades.csv'):
    trade_profit = round(report.end_balance - report.start_balance, 2)
    investment_profit_pct = round((trade_profit/report.investment_peak), 4) * 100
    trade_profit_log_pct = np.log(report.end_balance/report.start_balance) * 100  # R=ln(Vf/Vi)t×100
    with open(file_path, mode='a') as trades_file:
        trades_csv = csv.writer(trades_file, delimiter=',')
        trades_csv.writerow([
            report.id,
            # report.strategy_name,
            report.start_time.strftime("%m-%d-%Y, %H:%M:%S"),
            report.end_time.strftime("%m-%d-%Y, %H:%M:%S"),
            f'{round((report.end_time - report.start_time).total_seconds()/60, 3)} minutes',
            round(report.buy_strike_price, 2),
            round(report.sell_strike_price, 2),
            round(report.buy_quantity, 2),
            round(report.start_balance, 2),
            round(report.end_balance, 2),
            f'{investment_profit_pct}%',
            f'{round(trade_profit_log_pct, 7)}%',
            f'${trade_profit}',
            report.buy_time_indicators,
            report.notes])


def log_HTF_trade(report: TradeCycleReport):
    with open('traderplus/HFT_trades.csv', mode='a') as trades_file:
        trades_csv = csv.writer(trades_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        start_time = report.start_time
        end_time = report.end_time
        trade_duration = round((report.end_time - report.start_time).seconds / 60, 2)
        start_balance = report.start_balance
        end_balance = report.end_balance
        trade_profit = report.end_balance - report.start_balance
        num_of_buys = report.num_of_times_dca_used
        trades_csv.writerow([start_time, end_time, trade_duration, num_of_buys, start_balance, end_balance, trade_profit, report.notes])


def log_live_price(ticker_data: Any, file_path: str):
    with open(file_path, mode='a') as file:
            _csv = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            symbol = ticker_data['symbol']
            timestamp = datetime.now()
            priceChange = ticker_data['priceChange']
            priceChangePercent = ticker_data['priceChangePercent']
            weightedAvgPrice = ticker_data['weightedAvgPrice']
            prevClosePrice = ticker_data['prevClosePrice']
            lastPrice = ticker_data['lastPrice']
            lastQty = ticker_data['lastQty']
            bidPrice = ticker_data['bidPrice']
            bidQty = ticker_data['bidQty']
            askPrice = ticker_data['askPrice']
            askQty = ticker_data['askQty']
            openPrice = ticker_data['openPrice']
            highPrice = ticker_data['highPrice']
            lowPrice = ticker_data['lowPrice']
            volume = ticker_data['volume']
            _csv.writerow([symbol, timestamp, priceChange, priceChangePercent, weightedAvgPrice, prevClosePrice, lastPrice, lastQty, bidPrice, bidQty, askPrice, askQty, openPrice, highPrice, lowPrice, volume])


def log_vhft_action(action: Any, file_path: str):
    with open(file_path, mode='a') as file:
            _csv = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            price = action.price
            timestamp = action.timestamp
            type = action.type
            profit = action.profit
            quantity = action.quantity
            buy_time_indicators = action.buy_time_indicators
            _csv.writerow([type, timestamp, quantity, price, profit, buy_time_indicators])


def wait_til_candle_close(client: Client, symbol: str, candle_interval: str):
    time.sleep(2)
    now = datetime.now()
    current_candle_closetime = datetime.fromtimestamp(client.get_klines(symbol=symbol, interval=candle_interval)[-1][6]/1000)
    if current_candle_closetime > now:
        time_left = current_candle_closetime - now
        print(f'candle closes at {current_candle_closetime} in {round(time_left.seconds/60, 2)} minutes. Waiting...\n')
        time.sleep(time_left.seconds)
    else:
        time.sleep(1)
        return

    
def get_symbol_info(client: Client, symbol: str) -> SymbolParams:
    params = SymbolParams()
    info = client.get_symbol_info(symbol=symbol)
    params.base_asset_precision = float(info['baseAssetPrecision'])
    for filter in info['filters']:
        if filter['filterType'] == 'MIN_NOTIONAL' or filter['filterType'] == 'MIN_NOTIONAL':
            params.min_notional = float(filter['minNotional'])

        if filter['filterType'] == 'LOT_SIZE':
            params.min_lot_size = float(filter['minQty'])

    return params


def sigmoid(x: float):
  return 1 / (1 + math.exp(-x))


async def get_live_price(client: Client, symbol: str):
    bsm = BinanceSocketManager(client)
    socket = bsm.trade_socket(symbol)

    try:
        await socket.__aenter__()
        msg = await socket.recv()
        live_price = float(msg['p'])
        return live_price
    except BaseException as e:
        print(f'            failed to get price from socket: {e.args[0]}')


def synchronize_time_server():
    try:
        import ntplib
        client = ntplib.NTPClient()
        response = client.request('pool.ntp.org')
        os.system('date ' + time.strftime('%m%d%H%M%Y.%S',time.localtime(response.tx_time)))
    except:
        print('Could not sync with time server.')


def add_rsi(df: pd.DataFrame, length: int) -> pd.DataFrame:
    df[f'rsi_{length}'] = ta.rsi(df['Close'], length=length)
    return df


def add_ema(df: pd.DataFrame, length: int) -> pd.DataFrame:
    df[f'ema_{length}'] = ta.ema(df['Close'], length=length)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    macd = ta.macd(df['Close'])
    df['macd_line'] = macd['MACD_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    return df


def add_bollinger_bands(df: pd.DataFrame, length: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    bb = ta.bbands(df['Close'], length=length, std=std_dev)
    df['bb_upper'] = bb[f'BBU_{length}_{std_dev}']
    df['bb_middle'] = bb[f'BBM_{length}_{std_dev}']
    df['bb_lower'] = bb[f'BBL_{length}_{std_dev}']

    # derived features
    std = df['Close'].rolling(window=length, min_periods=1).std(ddof=0).fillna(0.0)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['percent_b'] = (df['Close'] - df['bb_lower']) / (df['bb_width'].replace({0: np.nan}))
    df['z_bb'] = (df['Close'] - df['bb_middle']) / (std.replace({0: np.nan}))
    return df


def get_balances_snapshot(client: Client, quote_asset: str, base_asset: str) -> Balances:
    whole_wallet = client.get_account()
    all_balances_dict = {balance['asset']: balance for balance in whole_wallet['balances'] if balance['asset'] == base_asset or balance['asset'] == quote_asset}
    free_quote_balance = float(all_balances_dict[quote_asset]['free'])
    locked_quote_balance = float(all_balances_dict[quote_asset]['locked'])
    free_base_balance = float(all_balances_dict[base_asset]['free'])
    locked_base_balance = float(all_balances_dict[base_asset]['locked'])
    b = Balances(free_base_balance=free_base_balance, locked_base_balance=locked_base_balance, free_quote_balance=free_quote_balance, locked_quote_balance=locked_quote_balance)
    return b


def get_instant_notional_minimum(client: Client, symbol: str, price: float) -> float:
    symbol_filter_dict = {filter['filterType']: filter for filter in client.get_symbol_info(symbol=symbol)['filters']}
    if 'MIN_NOTIONAL' in symbol_filter_dict:
            notional_key = 'MIN_NOTIONAL' 
    else: 
        notional_key = 'NOTIONAL'
    notional_minimum = float(symbol_filter_dict[notional_key]['minNotional'])/price
    return notional_minimum


def connection_is_good(client: Client, wait_until_true = True) -> bool:
    while True:
        try:
            client.get_server_time()
            return True
        except BaseException as e:
            print(msg=f'connection error: not getting server time: {e.args}')
            if not wait_until_true:
                return False
            time.sleep(3)