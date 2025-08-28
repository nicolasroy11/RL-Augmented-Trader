import uuid
from django.core.management.base import BaseCommand
from binance.enums import KLINE_INTERVAL_1MINUTE
from services.core.ML.configurations.fixture_config import CONFIG_UUIDS, DEFAULT_FEATURE_SET_ID
from services.core.models import (
    EMA,
    MACD,
    BaseObservationSet,
    RSI,
    BollingerBands,
    FeatureSet,
    DerivedFeature,
    DerivedfeatureSetMapping,
    RunConfiguration,
)

short_ema_length = 7
mid_ema_length = 12
long_ema_length = 30
xlong_ema_length = 150
short_rsi_length = 5
mid_rsi_length = 7
long_rsi_length = 12
bollinger_length = 14


DERIVED_FEATURE_UUIDS = {
    
    # indicator derived
    "short_gt_mid": "d1111111-1111-1111-1111-111111111111",
    "mid_gt_long": "d2222222-2222-2222-2222-222222222222",
    "all_trend_up": "d3333333-3333-3333-3333-333333333333",
    "price_gt_long": "d4444444-4444-4444-4444-444444444444",
    "breakout_high": "d5555555-5555-5555-5555-555555555555",
    "bb_squeeze": "d6666666-6666-6666-6666-666666666666",
    "ema_slope_sign": "d7777777-7777-7777-7777-777777777777",
    "dist_from_short_ema": "d8888888-8888-8888-8888-888888888888",
    "price_above_upper_bb": "d9999999-9999-9999-9999-999999999999",
    "price_below_lower_bb": "da111111-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "price_vs_middle_bb": "db222222-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    "bb_width": "dc333333-cccc-cccc-cccc-cccccccccccc",
    "bb_percent_b": "dd444444-dddd-dddd-dddd-dddddddddddd",

    # Position features
    "position_exists": "de555555-eeee-eeee-eeee-eeeeeeeeeeee",
    "position_relative_entry_price": "df666666-ffff-ffff-ffff-ffffffffffff",
    "position_unrealized_pnl": "e0777777-0000-0000-0000-000000000000",
}


class Command(BaseCommand):
    help = "Load fixtures"

    def handle(self, *args, **options):
        # BaseObservationSet
        base_obs_set, _ = BaseObservationSet.objects.get_or_create(
            id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            defaults={
                "name": "default",
                "candle_interval": KLINE_INTERVAL_1MINUTE,
            },
        )

        # RSI
        rsi_lengths = [
            (short_rsi_length, "22222222-2222-2222-2222-222222222222"),
            (mid_rsi_length, "55555555-5555-5555-5555-555555555555"),
            (long_rsi_length, "66666666-6666-6666-6666-666666666666"),
        ]
        for length, rsi_id in rsi_lengths:
            RSI.objects.get_or_create(
                id=uuid.UUID(rsi_id),
                defaults={
                    "length": length,
                    "is_sequence": True,
                    "base_observation_set_id": base_obs_set.id,
                },
            )

        # EMA
        ema_lengths = [
            (short_ema_length, "77777777-7777-7777-7777-777777777777"),
            (mid_ema_length, "88888888-8888-8888-8888-888888888888"),
            (long_ema_length, "99999999-9999-9999-9999-999999999999"),
            (xlong_ema_length, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        ]
        for length, ema_id in ema_lengths:
            EMA.objects.get_or_create(
                id=uuid.UUID(ema_id),
                defaults={
                    "length": length,
                    "is_sequence": True,
                    "base_observation_set_id": base_obs_set.id,
                },
            )

        # MACD
        MACD.objects.get_or_create(
            id=uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            defaults={
                "fast": 12,
                "slow": 26,
                "signal": 9,
                "is_sequence": True,
                "base_observation_set_id": base_obs_set.id,
            },
        )

        # BollingerBands
        BollingerBands.objects.get_or_create(
            id=uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
            defaults={
                "length": 20,
                "std_dev": 2.0,
                "is_sequence": True,
                "base_observation_set_id": base_obs_set.id,
            },
        )

        # FeatureSet
        feature_set, _ = FeatureSet.objects.get_or_create(
            id=DEFAULT_FEATURE_SET_ID,
            defaults={
                "name": "default",
                "base_observation_set_id": base_obs_set.id,
                "window_length": 150,
            },
        )

        # DerivedFeature and Mapping
        for method_name, feature_id in DERIVED_FEATURE_UUIDS.items():
            derived_feature, _ = DerivedFeature.objects.get_or_create(
                id=uuid.UUID(feature_id),
                defaults={
                    "method_name": method_name,
                    "is_sequence": False,
                },
            )

            DerivedfeatureSetMapping.objects.get_or_create(
                feature_set_id=feature_set.id,
                derived_feature_id=uuid.UUID(feature_id),
            )

        self.stdout.write(self.style.SUCCESS("Fixture data created successfully!"))


        # RunConfiguration
        for config_class, config_uuid in CONFIG_UUIDS.items():
            RunConfiguration.objects.get_or_create(
                id=config_uuid,
                defaults={
                    "name": config_class.__name__,
                    "description": config_class.__str__(),
                },
            )

        self.stdout.write(self.style.SUCCESS("RunConfiguration fixtures created successfully!"))
