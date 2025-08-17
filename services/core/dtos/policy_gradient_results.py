from typing import List
from rest_framework import serializers
from services.decorators.decorators.dto_class_decorator import Dto


@Dto()
class PolicyGradientResults():
    all_episode_final_pnls: List[float]
    class Serializer(serializers.Serializer):
        all_episode_final_pnls = serializers.ListField(child=serializers.FloatField())
        # buy_and_hold_pnl = serializers.FloatField()
