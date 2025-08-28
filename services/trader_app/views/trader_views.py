from django.core.handlers.wsgi import WSGIRequest
from django.http import JsonResponse
from services.core.dtos.full_single_long_cycle_dto import FullSingleLongCycleDto
from services.decorators.decorators.view_decorator import View
from services.decorators.decorators.view_class_decorator import ViewClass
from services.core.ML.configurations.PPO_flattened_history.trade import TraderRepository

trader_repo = TraderRepository()


@ViewClass(
    url='traders/'
)
class TraderViews:

    class Meta:
        app_label = 'services.trader_app'
        label = 'traders'


    @View(
        path='run_single_buy_ppo_trader',
        http_method='GET',
        return_type=FullSingleLongCycleDto.Serializer(many=True),
        description='Runs N cycles of single blocking buy trader',
        include_in_swagger=True
    )
    def run_single_buy_ppo_trader(req: WSGIRequest):
        feature_set = req.GET.get('feature_set')
        def exec():
            results = trader_repo.run_single_buy_ppo_trader(feature_set_name=feature_set)
            dto = FullSingleLongCycleDto.Serializer(results, many=True).data
            return JsonResponse(dto, safe=False)
        return exec()
