from django.db import models

DUCKDB_TRANSFERS_SCHEMA = 'duckdb_transfers'


class BTCFDUSDTick(models.Model):
    class Meta:
        db_table = f'"{DUCKDB_TRANSFERS_SCHEMA}"."btcfdusd_ticks"'

    def __str__(self):
        return f"{self.timestamp} | {self.price}"

    timestamp = models.DateTimeField(unique=True)
    price = models.FloatField()
    rsi_5 = models.FloatField()
    rsi_7 = models.FloatField()
    rsi_12 = models.FloatField()
    ema_short = models.FloatField()
    ema_mid = models.FloatField()
    ema_long = models.FloatField()
    ema_xlong = models.FloatField()
    macd_line = models.FloatField()
    macd_hist = models.FloatField()
    macd_signal = models.FloatField()
    bb_upper = models.FloatField()
    bb_middle = models.FloatField()
    bb_lower = models.FloatField()


class BTCFDUSDData(models.Model):
    class Meta:
        db_table = f'"public"."btcfdusd_data"'

    def __str__(self):
        return f"{self.timestamp} | {self.price}"

    timestamp = models.DateTimeField(unique=True, null=False)
    price = models.FloatField(null=True)
    rsi_5 = models.FloatField(null=True)
    rsi_7 = models.FloatField(null=True)
    rsi_12 = models.FloatField(null=True)
    ema_short = models.FloatField(null=True)
    ema_mid = models.FloatField(null=True)
    ema_long = models.FloatField(null=True)
    ema_xlong = models.FloatField(null=True)
    macd_line = models.FloatField(null=True)
    macd_hist = models.FloatField(null=True)
    macd_signal = models.FloatField(null=True)
    bb_upper = models.FloatField(null=True)
    bb_middle = models.FloatField(null=True)
    bb_lower = models.FloatField(null=True)

