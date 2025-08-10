from django.apps import AppConfig
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from db.data_store import DataStore
from django.conf import settings

data_store: DataStore = None

class DataApp(AppConfig):
    name = "data_app"
    verbose_name = "data app"

    def ready(self):
        db_path = settings.DUCKDB_FILE_PATH if hasattr(settings, 'DUCKDB_FILE_PATH') else 'data/ticks.duckdb'
        data_store = DataStore(db_path=db_path, readonly=False)
        data_store.create_table()
        scheduler = BackgroundScheduler(timezone=pytz.UTC)
        scheduler.add_job(data_store.single_poll_and_store, id='single_poll_and_store', trigger='interval', seconds=5)
        scheduler.start()