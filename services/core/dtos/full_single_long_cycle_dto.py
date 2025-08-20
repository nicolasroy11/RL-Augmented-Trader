from typing import List
from rest_framework import serializers
from services.decorators.decorators.dto_class_decorator import Dto


@Dto()
class FullSingleLongCycleDto:
    start_time = None
    end_time = None
    position_size = 0.0
    entry_price = 0.0
    exit_price = 0.0
    initial_quote_balance = 0.0
    exit_quote_balance = 0.0
    entry_buy_probability = 0.0
    exit_sell_probability = 0.0
    realized_pnl = 0.0

    class Serializer(serializers.Serializer):
        start_time = serializers.DateTimeField()
        end_time = serializers.DateTimeField()
        position_size = serializers.FloatField()
        entry_price = serializers.FloatField()
        exit_price = serializers.FloatField()
        initial_quote_balance = serializers.FloatField()
        exit_quote_balance = serializers.FloatField()
        entry_buy_probability = serializers.FloatField()
        exit_sell_probability = serializers.FloatField()
        realized_pnl = serializers.FloatField()