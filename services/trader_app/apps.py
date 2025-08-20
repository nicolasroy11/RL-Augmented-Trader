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
