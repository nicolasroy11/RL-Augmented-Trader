from django.apps import AppConfig
import runtime_settings

class TradersConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "services.trader_app"

    def ready(self):
        client_test = runtime_settings.write_client.create_test_order(
            symbol='ETHUSDT',
            side='BUY',
            type='LIMIT',
            timeInForce='GTC',
            quantity=0.01,
            price='2000.00'
        )
        if client_test != {}:
            raise RuntimeError()
