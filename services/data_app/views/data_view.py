from django.core.handlers.wsgi import WSGIRequest
from django.http import JsonResponse
from django.views import View
from services.data_app.dtos.tick_dto import TickDto
from services.decorators.decorators.http_method_decorator import Http
from apps import data_store


@View(
    url='data'
)
class Ticks:

    class Meta:
        app_label = 'services.data_app'
        label = 'ticks'

    @Http(
        path='',
        http_method='GET',
        return_type=TickDto.Serializer(many=True),
        description='Get all ticks'
    )
    def get_all(req: WSGIRequest):
        def exec():
            ticks = data_store.get_all_tick_data()
            dto = TickDto.Serializer(ticks, many=True).data
            return JsonResponse(dto, safe=False)
        return exec()