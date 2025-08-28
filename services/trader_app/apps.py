from typing import List
from django.apps import AppConfig
import runtime_settings


class TradersConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "services.trader_app"
