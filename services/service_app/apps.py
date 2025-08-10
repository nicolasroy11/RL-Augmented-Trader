from django.apps import AppConfig
import pytz


class TraderPlus(AppConfig):
    name = "service_app"
    verbose_name = "service app"

    async def ready(self):
        scheduler = BackgroundScheduler(timezone=pytz.UTC)
        # scheduler.add_job(scheduler_get_daily_generation_benchmark_grid_forecast,
        #                     'cron', id='Scheduler_get_daily_generation_benchmark_grid_forecast', day='*', misfire_grace_time=15*60)