from django.apps import AppConfig
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from db.data_store import DataStore
from django.conf import settings


class DataApp(AppConfig):
    name = "services.data_app"
    verbose_name = "data app"

    def ready(self):
        from services.data_app.repositories.data_repository import DataRepository
        data_repo = DataRepository()
        scheduler = BackgroundScheduler(timezone=pytz.UTC)
        scheduler.add_job(data_repo.single_poll_and_store, id='single_poll_and_store', trigger='interval', seconds=5)
        scheduler.start()