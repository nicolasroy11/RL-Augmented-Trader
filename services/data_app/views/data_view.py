from django.core.handlers.wsgi import WSGIRequest
from django.http import JsonResponse
from services.data_app.dtos.tick_dto import TickDto
from services.decorators.decorators.view_decorator import View
from services.decorators.decorators.view_class_decorator import ViewClass
from services.data_app.apps import data_store



@ViewClass(
    url='data/'
)
class Ticks:

    class Meta:
        app_label = 'services.data_app'
        label = 'ticks'

    @View(
        path='get_all_ticks',
        http_method='GET',
        return_type=TickDto.Serializer(many=True),
        description='Get all ticks'
    )
    def get_all(req: WSGIRequest):
        def exec():
            ticks = data_store.get_all_tick_data().to_dict(orient='records')
            print(f"Fetched {len(ticks)} ticks from the database. Approximately {len(ticks) * 5 / 60:.2f} minutes of data.")
            dto = TickDto.Serializer(ticks, many=True).data
            return JsonResponse(dto, safe=False)
        return exec()