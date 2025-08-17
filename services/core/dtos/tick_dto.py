from rest_framework import serializers
from db.data_store import Tick
from services.decorators.decorators.dto_class_decorator import Dto


@Dto()
class TickDto(Tick):

    class Serializer(serializers.Serializer):
        price = serializers.FloatField()
        timestamp = serializers.DateTimeField()
        price = serializers.FloatField()
        rsi_5 = serializers.FloatField()
        rsi_7 = serializers.FloatField()
        rsi_12 = serializers.FloatField()
        ema_short = serializers.FloatField()
        ema_mid = serializers.FloatField()
        ema_long = serializers.FloatField()
        ema_xlong = serializers.FloatField()
        macd_line = serializers.FloatField()
        macd_hist = serializers.FloatField()
        macd_signal = serializers.FloatField()
        bb_upper = serializers.FloatField()
        bb_middle = serializers.FloatField()
        bb_lower = serializers.FloatField()
        bb_width = serializers.FloatField()
        percent_b = serializers.FloatField()
        z_bb = serializers.FloatField()

