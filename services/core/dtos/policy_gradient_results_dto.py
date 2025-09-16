from typing import List, Optional
from rest_framework import serializers
from services.decorators.decorators.dto_class_decorator import Dto


@Dto()
class EpisodeResultsDto:
    episode_number: int
    final_pnl: float
    episode_pnls: List[float]
    running_max: List[float]
    drawdowns: List[float]
    action_probs: List[List[float]]
    max_drawdown: float
    buy_and_hold_pnl: float
    sharpe_ratio: float
    prices: List[float]

    validation_pnl: float
    validation_sharpe: float
    validation_max_drawdown: float
    validation_trades: int
    model_path: Optional[str]


    class Serializer(serializers.Serializer):
        episode_number = serializers.IntegerField()
        final_pnl = serializers.FloatField()
        episode_pnls = serializers.ListField(child=serializers.FloatField())
        running_max = serializers.ListField(child=serializers.FloatField())
        drawdowns = serializers.ListField(child=serializers.FloatField())
        max_drawdown = serializers.FloatField()
        buy_and_hold_pnl = serializers.FloatField()
        sharpe_ratio = serializers.FloatField()


@Dto()
class PPOTCNTrainingResults:
    episode_results: List[EpisodeResultsDto]
    holdout_pnl: float
    holdout_sharpe: float
    holdout_max_drawdown: float
    holdout_trades: int
    best_model_path: Optional[str]

    class Serializer(serializers.Serializer):
        episode_results = serializers.ListField(child=EpisodeResultsDto.Serializer())

