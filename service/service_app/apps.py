from django.apps import AppConfig


class TraderPlus(AppConfig):
    name = "service_app"
    verbose_name = "service app"

    async def ready(self):
        pass