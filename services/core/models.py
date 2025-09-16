from typing import List
import uuid
from django.db import models
from binance.enums import SIDE_BUY, SIDE_SELL, KLINE_INTERVAL_1MINUTE, KLINE_INTERVAL_15MINUTE
import pandas as pd
import runtime_settings


WINDOW_LENGTH = runtime_settings.DATA_TICKS_WINDOW_LENGTH

class TrainingFields(models.Model):
    class Meta:
        abstract = True

    price = models.FloatField(null=True)
    rsi_1 = models.FloatField(null=True)
    rsi_2 = models.FloatField(null=True)
    rsi_3 = models.FloatField(null=True)
    rsi_4 = models.FloatField(null=True)
    ema_1 = models.FloatField(null=True)
    ema_2 = models.FloatField(null=True)
    ema_3 = models.FloatField(null=True)
    ema_4 = models.FloatField(null=True)
    macd_line = models.FloatField(null=True)
    macd_hist = models.FloatField(null=True)
    macd_signal = models.FloatField(null=True)
    bb_upper = models.FloatField(null=True)
    bb_middle = models.FloatField(null=True)
    bb_lower = models.FloatField(null=True)
    supertrend_line = models.FloatField(null=True)
    direction_flag = models.FloatField(null=True)
    long_stop_line = models.FloatField(null=True)
    short_stop_line = models.FloatField(null=True)
        
    @staticmethod
    def get_fields():
        return {field.name for field in TrainingFields._meta.get_fields() if field.concrete and not field.many_to_many}


class DataRun(models.Model):
    class Meta:
        db_table = f'"public"."data_runs"'

    def __str__(self):
        return f"{self.id}_{self.com_or_us}_{self.is_testnet}_{self.base_asset}_{self.quote_asset}"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, verbose_name='data_run_id', null=False)
    com_or_us = models.CharField(max_length=3, choices=[('com', 'com'), ('us', 'us')], null=False)
    is_testnet = models.BooleanField(null=False)
    is_futures = models.BooleanField(null=False)
    base_asset = models.CharField(max_length=7, null=False)
    quote_asset = models.CharField(max_length=7, null=False)
    window_length = models.IntegerField(null=False, default=runtime_settings.DATA_TICKS_WINDOW_LENGTH)
    run_type = models.CharField(max_length=8, choices=[('training', 'training'), ('trading', 'trading')], null=True)


class TickData(TrainingFields):
    class Meta:
        db_table = f'"public"."tick_data"'

    def __str__(self):
        return f"{self.price}"
    
    @staticmethod
    def remove_non_training_fields_in_df(df: pd.DataFrame):
        """
        Keep only the columns in the DataFrame that match fields of the given Django model.
        """
        model_fields = {field.name for field in TrainingFields._meta.get_fields() if field.concrete and not field.many_to_many}
        valid_columns = [col for col in df.columns if col in model_fields]
        return df[valid_columns]

    @staticmethod
    def list_to_env_dataframe(tick_list: List['TickData']) -> pd.DataFrame:
        ticks_df = pd.DataFrame.from_records([t.__dict__ for t in tick_list])
        ticks_df = ticks_df.drop(columns=["_state"], errors="ignore")
        if len(tick_list) < WINDOW_LENGTH:
            print(f'interpolating {WINDOW_LENGTH - len(tick_list)} ticks')
            ticks_df = TickData.get_interpolated_window_df(ticks_df)
        return ticks_df.reset_index(drop=True)
    
    @staticmethod
    def get_interpolated_window_df(ticks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce a WINDOW_LENGTH DataFrame of TickData at 5-second intervals.
        - Linear interpolation for numeric fields.
        - Handles millisecond timestamps correctly.
        - Ensures exact WINDOW_LENGTH rows via backfilling at start if necessary.
        """

        df = ticks_df.copy().sort_values("timestamp")

        # Determine last timestamp and window start
        last_ts = df["timestamp"].max().ceil("5s")
        first_ts = last_ts - pd.Timedelta(seconds=5 * (WINDOW_LENGTH - 1))

        # Create the fixed 5-second grid
        grid_df = pd.DataFrame({"timestamp": pd.date_range(start=first_ts, end=last_ts, freq="5s")})

        # Merge original data onto the grid using nearest previous timestamps
        merged = pd.merge_asof(grid_df, df, on="timestamp", direction="backward")

        # Interpolate numeric columns linearly
        numeric_cols = merged.select_dtypes(include="number").columns
        merged[numeric_cols] = merged[numeric_cols].interpolate(method="linear").bfill()

        return merged

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, verbose_name='tick_data_id', null=False, db_index=True)
    timestamp = models.DateTimeField(unique=True, db_index=True)
    data_run = models.ForeignKey(DataRun, on_delete=models.CASCADE, null=False)


class TickProbabilities(models.Model):
    class Meta:
        db_table = f'"public"."tick_probabilities"'

    tick_data = models.ForeignKey(TickData, on_delete=models.CASCADE, null=False)
    buy_prob = models.FloatField(null=True)
    hold_prob = models.FloatField(null=True)
    sell_prob = models.FloatField(null=True)


class BaseObservationSet(models.Model):
    class Meta:
        db_table = '"public"."base_observation_sets"'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    name = models.CharField(max_length=40, null=True)
    candle_interval = models.CharField(max_length=10,choices=[(KLINE_INTERVAL_1MINUTE, KLINE_INTERVAL_1MINUTE),(KLINE_INTERVAL_15MINUTE, KLINE_INTERVAL_15MINUTE)])

    def __str__(self):
        return self.name


class RSI(models.Model):
    class Meta:
        db_table = '"public"."rsi"'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    length = models.IntegerField()
    is_sequence = models.BooleanField()
    base_observation_set = models.ForeignKey(BaseObservationSet, on_delete=models.CASCADE, related_name="rsis")

    def __str__(self):
        return f"rsi_{self.length}"
    
class EMA(models.Model):
    class Meta:
        db_table = '"public"."ema"'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    length = models.IntegerField()
    is_sequence = models.BooleanField()
    base_observation_set = models.ForeignKey(BaseObservationSet, on_delete=models.CASCADE, related_name="emas")

    def __str__(self):
        return f"ema_{self.length}"
    
class MACD(models.Model):
    class Meta:
        db_table = '"public"."macd"'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    fast = models.IntegerField()
    slow = models.IntegerField()
    signal = models.IntegerField()
    is_sequence = models.BooleanField()
    base_observation_set = models.ForeignKey(BaseObservationSet, on_delete=models.CASCADE, related_name="macds")

    def __str__(self):
        return f"macd_{self.fast}_{self.slow}_{self.signal}'"
    
class BollingerBands(models.Model):
    class Meta:
        db_table = '"public"."bollinger_bands"'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    length = models.IntegerField()
    std_dev = models.FloatField()
    is_sequence = models.BooleanField()
    base_observation_set = models.ForeignKey(BaseObservationSet, on_delete=models.CASCADE, related_name="bbands")

    def __str__(self):
        return f"bband_{self.length}_{self.std_dev}'"
    

class SuperTrend(models.Model):
    class Meta:
        db_table = '"public"."supertrends"'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    length = models.IntegerField()
    multiplier = models.FloatField()
    is_sequence = models.BooleanField()
    base_observation_set = models.ForeignKey(BaseObservationSet, on_delete=models.CASCADE, related_name="strends")

    def __str__(self):
        return f"strend_{self.length}_{self.multiplier}'"


class FeatureSet(models.Model):
    class Meta:
        db_table = '"public"."feature_sets"'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    name = models.CharField(max_length=40)
    base_observation_set = models.ForeignKey(BaseObservationSet, on_delete=models.CASCADE, related_name="feature_sets")
    window_length = models.IntegerField()

    def __str__(self):
        return self.name
    
    def get_feature_vector_size(self):
        vector_size = 0
        base_set = self.base_observation_set
        vector_size += self.window_length # TODO: because of the price column, include the price in a dynamic way

        rsi: RSI
        for rsi in base_set.rsis.all():
            if rsi.is_sequence:
                vector_size += self.window_length
            else:
                vector_size += 1

        ema: EMA
        for ema in base_set.emas.all():
            if ema.is_sequence:
                vector_size += self.window_length
            else:
                vector_size += 1

        macd: MACD
        for macd in base_set.macds.all():
            if macd.is_sequence:
                vector_size += (self.window_length * 3)
            else:
                vector_size += 3

        bband: BollingerBands
        for bband in base_set.bbands.all():
            if bband.is_sequence:
                vector_size += (self.window_length * 3)
            else:
                vector_size += 3

        strend: SuperTrend
        for strend in base_set.strends.all():
            if strend.is_sequence:
                vector_size += (self.window_length * 3)
            else:
                vector_size += 3

        derived_feature_mappings = list(DerivedfeatureSetMapping.objects.filter(feature_set_id=self.id))
        m: DerivedfeatureSetMapping
        for m in derived_feature_mappings:
            if m.derived_feature.is_sequence:
                vector_size += self.window_length
            else:
                vector_size += 1

        return vector_size
    

class DerivedFeature(models.Model):
    class Meta:
        db_table = '"public"."derived_features"'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    method_name = models.CharField(max_length=40)
    is_sequence = models.BooleanField()

    def __str__(self):
        return self.method_name


class DerivedfeatureSetMapping(models.Model):
    class Meta:
        db_table = '"public"."derived_feature_mappings"'

    feature_set = models.ForeignKey(FeatureSet, on_delete=models.CASCADE, related_name="mappings")
    derived_feature = models.ForeignKey(DerivedFeature, on_delete=models.CASCADE, related_name="mapped_sets")

    def __str__(self):
        return f"{self.feature_set.name} - {self.derived_feature.method_name}"


class RunConfiguration(models.Model):
    class Meta:
        db_table = f'"public"."run_configurations"'

    id = models.UUIDField(primary_key=True, editable=False)
    name = models.CharField(max_length=150, null=True)
    description = models.CharField(max_length=250, null=True)
    blocking = models.BooleanField(null=False)


class TradingSession(models.Model):
    class Meta:
        db_table = f'"public"."trading_sessions"'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, verbose_name='trading_session_id', null=False)
    data_run = models.ForeignKey(DataRun, on_delete=models.DO_NOTHING, null=True)
    feature_set = models.ForeignKey(FeatureSet, on_delete=models.CASCADE, null=False)
    run_configuration = models.ForeignKey(RunConfiguration, on_delete=models.CASCADE, null=False)


class TrainingSession(models.Model):
    class Meta:
        db_table = f'"public"."training_sessions"'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, verbose_name='training_session_id', null=False)
    num_episodes = models.IntegerField()
    data_runs = models.ManyToManyField(DataRun)
    feature_set = models.ForeignKey(FeatureSet, on_delete=models.CASCADE, null=False)
    run_configuration = models.ForeignKey(RunConfiguration, on_delete=models.CASCADE, null=True)


class Transaction(models.Model):
    class Meta:
        db_table = f'"public"."transactions"'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, verbose_name='single_buy_cycle_id', null=False)
    side = models.CharField(max_length=8, choices=[(SIDE_BUY, SIDE_BUY), (SIDE_SELL, SIDE_SELL)], null=True)
    tick_data = models.ForeignKey(TickData, on_delete=models.DO_NOTHING, null=False)
    trading_session = models.ForeignKey(TradingSession, on_delete=models.CASCADE, null=False)
    strike_price = models.FloatField()
    base_amount = models.FloatField()


class MLModel(models.Model):
    class Meta:
        db_table = f'"public"."ml_models"'
        db_table_comment = 'The id on this table will be the name of the pickle file.'

    def __str__(self):
        return f"{self.run_configuration.str()}"
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    feature_set = models.ForeignKey(FeatureSet, on_delete=models.CASCADE, null=False)
    run_configuration = models.ForeignKey(RunConfiguration, on_delete=models.DO_NOTHING, null=False)


class Hyperparameter(models.Model):
    class Meta:
        db_table = f'"public"."hyperparameters"'
        unique_together = ("ml_model", "key")  # each param unique per model

    def __str__(self):
        return f"{self.ml_model.id}: {self.key}={self.value}"
    
    ml_model = models.ForeignKey(MLModel, related_name="hyperparameters", on_delete=models.CASCADE)
    key = models.CharField(max_length=100)   # e.g. "clip_ratio"
    value = models.CharField(max_length=255) # always stored as string

    