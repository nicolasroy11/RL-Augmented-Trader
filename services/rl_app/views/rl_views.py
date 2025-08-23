from django.core.handlers.wsgi import WSGIRequest
from django.http import JsonResponse
import runtime_settings
from services.core.dtos.policy_gradient_results_dto import PolicyGradientResultsDto
from services.decorators.decorators.view_decorator import View
from services.decorators.decorators.view_class_decorator import ViewClass
from services.rl_app.repositories.rl_repository import RLRepository

rl_repo = RLRepository()


@ViewClass(
    url='rl/'
)
class RLViews:

    class Meta:
        app_label = 'services.rl_app'
        label = 'rl'


    @View(
        path='run_policy_gradient',
        http_method='GET',
        return_type=PolicyGradientResultsDto.Serializer(),
        description='Run policy gradient algorithm and return results',
        include_in_swagger=True
    )
    def run_policy_gradient(req: WSGIRequest):
        def exec():
            results = rl_repo.run_policy_gradient(window_size=runtime_settings.DATA_TICKS_WINDOW_LENGTH, num_episodes=100)
            dto = PolicyGradientResultsDto.Serializer(results).data
            return JsonResponse(dto)
        return exec()


    @View(
        path='run_ppo',
        http_method='GET',
        return_type=PolicyGradientResultsDto.Serializer(),
        description='Run policy gradient algorithm and return results',
        include_in_swagger=True
    )
    def run_ppo(req: WSGIRequest):
        def exec():
            results = rl_repo.run_ppo(window_size=runtime_settings.DATA_TICKS_WINDOW_LENGTH, num_episodes=100)
            dto = PolicyGradientResultsDto.Serializer(results).data
            return JsonResponse(dto)
        return exec()
    