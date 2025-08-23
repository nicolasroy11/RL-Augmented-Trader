from django.apps import AppConfig
import pytz
from apscheduler.triggers.cron import CronTrigger
from apscheduler.schedulers.background import BackgroundScheduler


class DataApp(AppConfig):
    name = "services.data_app"
    verbose_name = "data app"
