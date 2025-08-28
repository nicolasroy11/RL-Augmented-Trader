from django.core.handlers.wsgi import WSGIRequest
from django.http import JsonResponse
import runtime_settings
from services.core.dtos.policy_gradient_results_dto import PolicyGradientResultsDto
from services.decorators.decorators.view_decorator import View
from services.decorators.decorators.view_class_decorator import ViewClass
from services.core.ML.configurations.PPO_flattened_history.train import RLRepository


@ViewClass(
    url='training/'
)
class RLViews:

    class Meta:
        app_label = 'services.trainer_app'
        label = 'training'


    @View(
        path='run_policy_gradient',
        http_method='GET',
        return_type=PolicyGradientResultsDto.Serializer(),
        description='Run policy gradient algorithm and return results',
        include_in_swagger=True
    )
    def run_policy_gradient(req: WSGIRequest):
        def exec():
            rl_repo = RLRepository()
            results = rl_repo.run_policy_gradient(window_size=runtime_settings.DATA_TICKS_WINDOW_LENGTH, num_episodes=100)
            dto = PolicyGradientResultsDto.Serializer(results).data
            return JsonResponse(dto)
        return exec()


    @View(
        path='run_ppo',
        http_method='POST',
        return_type=PolicyGradientResultsDto.Serializer(),
        description='Run policy gradient algorithm and return results',
        include_in_swagger=True
    )
    def run_ppo(req: WSGIRequest):
        feature_set = req.GET.get('feature_set')
        # num_episodes=100, gamma=0.99, lr=1e-4, clip_epsilon=0.2, ppo_epochs=4, batch_size=64
        def exec():
            rl_repo = RLRepository()
            results = rl_repo.run_ppo(window_size=runtime_settings.DATA_TICKS_WINDOW_LENGTH, num_episodes=100)
            dto = PolicyGradientResultsDto.Serializer(results).data
            return JsonResponse(dto)
        return exec()
    