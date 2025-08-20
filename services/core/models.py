from django.db import models

DUCKDB_TRANSFERS_SCHEMA = 'duckdb_transfers'


class CommonTickFields(models.Model):
    class Meta:
        abstract=True

    timestamp = models.DateTimeField(unique=True)
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


class BTCFDUSDTick(CommonTickFields):
    class Meta:
        db_table = f'"{DUCKDB_TRANSFERS_SCHEMA}"."btcfdusd_ticks"'

    def __str__(self):
        return f"{self.timestamp} | {self.price}"


class BTCFDUSDData(CommonTickFields):
    class Meta:
        db_table = f'"public"."btcfdusd_data"'

    def __str__(self):
        return f"{self.timestamp} | {self.price}"


class TraderRun(models.Model):
    class Meta:
        db_table = f'"public"."trader_run"'

    def __str__(self):
        return f"{self.created_at} | {self.com_or_us} | {self.is_testnet} | {self.base_asset} | {self.quote_asset}"

    id = models.UUIDField(primary_key=True, unique=True, verbose_name='trader_run_id', null=False)
    com_or_us = models.CharField(max_length=3, choices=[('com', 'com'), ('us', 'us')], null=False)
    is_testnet = models.BooleanField(null=False)
    base_asset = models.CharField(max_length=7, null=False)
    quote_asset = models.CharField(max_length=7, null=False)
    created_at = models.DateTimeField()


class SingleBuyCycle(models.Model):
    class Meta:
        db_table = f'"public"."single_buy_cyle"'

    def __str__(self):
        return f"{self.run_id} | {self.start_time} | {self.end_time}"

    id = models.UUIDField(primary_key=True,unique=True, verbose_name='single_buy_cycle_id', null=False)
    run_id = models.ForeignKey(TraderRun, on_delete=models.CASCADE)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()


class RunTickData(CommonTickFields):
    class Meta:
        db_table = f'"public"."run_tick_data"'

    def __str__(self):
        return f"{self.run_id} | {self.datetime}"

    id = models.UUIDField(primary_key=True, unique=True, verbose_name='run_tick_data_id', null=False)
    run_id = models.ForeignKey(TraderRun, on_delete=models.CASCADE)
    datetime = models.DateTimeField()
    buy_prob = models.FloatField()
    hold_prob = models.FloatField()
    sell_prob = models.FloatField()
