from django.core.handlers.wsgi import WSGIRequest
from django.http import JsonResponse
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
            results = rl_repo.run_policy_gradient(window_size=150, num_episodes=10)
            dto = PolicyGradientResultsDto.Serializer(results).data
            return JsonResponse(dto)
        return exec()
    
    