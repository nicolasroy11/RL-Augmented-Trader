
from django.core.handlers.wsgi import WSGIRequest
from django.http import JsonResponse
from urllib3 import HTTPResponse
from services.core.dtos.tick_dto import TickDto
from services.decorators.decorators.view_decorator import View
from services.decorators.decorators.view_class_decorator import ViewClass
import pytz
from apscheduler.triggers.cron import CronTrigger
from apscheduler.schedulers.background import BackgroundScheduler
from services.data_app.repositories.data_repository import DataRepository
data_repo = DataRepository()


@ViewClass(
    url='data/'
)
class Ticks:

    class Meta:
        app_label = 'services.data_app'
        label = 'ticks'

    @View(
        path='start_data_collection',
        http_method='POST',
        return_type=TickDto.Serializer(many=True),
        description='Get all ticks'
    )
    def start_data_collection(req: WSGIRequest):
        def exec():
            scheduler = BackgroundScheduler(timezone=pytz.UTC)
            scheduler.add_job(
                data_repo.single_poll_and_store,
                trigger=CronTrigger(second='*/5'),
                id='single_poll_and_store',
                max_instances=1,
                coalesce=False,
            )
            scheduler.start()
            return HTTPResponse("Data collection started", status=200)
        return exec()

    @View(
        path='get_all_ticks',
        http_method='GET',
        return_type=TickDto.Serializer(many=True),
        description='Get all ticks'
    )
    def get_all(req: WSGIRequest):
        def exec():
            ticks = data_repo.get_ticks_from_duckdb()
            print(f"Fetched {len(ticks)} ticks from the database. Approximately {len(ticks) * 5 / 60:.2f} minutes of data.")
            dto = TickDto.Serializer(ticks, many=True).data
            return JsonResponse(dto, safe=False)
        return exec()
    

    @View(
        path='save_archives_to_pg',
        http_method='GET',
        return_type=TickDto.Serializer(many=True),
        description='Get all ticks'
    )
    def save_archives_to_pg(req: WSGIRequest):
        def exec():
            ticks = data_repo.save_ticks_to_pg()
            dto = TickDto.Serializer(ticks, many=True).data
            return JsonResponse(dto, safe=False)
        return exec()