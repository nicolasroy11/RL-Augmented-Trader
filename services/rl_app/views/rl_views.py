from django.core.handlers.wsgi import WSGIRequest
from django.http import HttpResponse, JsonResponse
from services.data_app.dtos.tick_dto import TickDto
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
        path='start_stochatic_rl',
        http_method='GET',
        return_type=TickDto.Serializer(many=True),
        description='Get all ticks'
    )
    def get_all(req: WSGIRequest):
        def exec():
            rl_repo.run_policy_gradient(window_size=150, num_episodes=100)
        return exec()
    