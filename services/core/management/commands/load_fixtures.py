import uuid
from django.core.management.base import BaseCommand
from binance.enums import KLINE_INTERVAL_1MINUTE, KLINE_INTERVAL_5MINUTE
from services.core.ML.configurations.fixture_config import CONFIG_UUIDS, DEFAULT_FEATURE_SET_ID, PPO_TCN_FUTURES_SET_ID
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
    SuperTrend,
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
    "ema_1_slope_sign": "d7777777-7777-7777-7777-777777777777",
    "ema_2_slope_sign": "dd777777-7777-7777-7777-777777777777",
    "ema_3_slope_sign": "ddd77777-7777-7777-7777-777777777777",
    "ema_4_slope_sign": "dddd7777-7777-7777-7777-777777777777",
    "dist_from_short_ema": "d8888888-8888-8888-8888-888888888888",
    "price_above_upper_bb": "d9999999-9999-9999-9999-999999999999",
    "price_below_lower_bb": "da111111-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "price_vs_middle_bb": "db222222-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    "bb_width": "dc333333-cccc-cccc-cccc-cccccccccccc",
    "bb_percent_b": "dd444444-dddd-dddd-dddd-dddddddddddd",
    "threshold_rsi_crossed_up_recent": "8b479921-4a26-4eec-a5dd-4bced0b470b2",
    "threshold_rsi_crossed_down_recent": "4f2e5123-b687-4178-a5c1-b050807413ba",
    "threshold_price_is_highest_in_window": "6df3b2cf-6fe3-43b5-bb81-adc20aa332e5",

    # Position features
    "position_exists": "de555555-eeee-eeee-eeee-eeeeeeeeeeee",
    "position_relative_entry_price": "df666666-ffff-ffff-ffff-ffffffffffff",
    "position_unrealized_pnl": "e0777777-0000-0000-0000-000000000000",
}


class Command(BaseCommand):
    help = "Load fixtures"

    def handle(self, *args, **options):


        #========================================= Default Observation Set ===============================================


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



        #======================================== OPP TCN Futures Observation Set ========================================


        tcn_5_min_obs_set, _ = BaseObservationSet.objects.get_or_create(
            id=uuid.UUID("14578f0f-1bd2-4468-92e2-e98690a68ca1"),
            defaults={
                "name": "tcn 5min observation set",
                "candle_interval": KLINE_INTERVAL_5MINUTE,
            },
        )

        # RSI
        rsi_lengths = [
            (7, "d148b67a-ac8b-43d3-9fb3-3437a03dac8b"),
            (14, "de292e0b-1fb2-4470-93bb-93efcebcb88a"),
            (30, "be58a0fe-d25f-4c5e-8aae-834867e22e6e"),
        ]
        for length, rsi_id in rsi_lengths:
            RSI.objects.get_or_create(
                id=uuid.UUID(rsi_id),
                defaults={
                    "length": length,
                    "is_sequence": True,
                    "base_observation_set_id": tcn_5_min_obs_set.id,
                },
            )

        # EMA
        ema_lengths = [
            (7, "30361c0f-13ba-42f1-89ca-b022f940168b"),
            (12, "39adbdd9-4c32-4b9d-af4b-bee9abb84317"),
            (50, "37e996a6-3599-46d1-ae0f-ee2d878f2efa"),
            (250, "fd9525be-a4fe-474f-9f04-b1a9a550c35e"),
        ]
        for length, ema_id in ema_lengths:
            EMA.objects.get_or_create(
                id=uuid.UUID(ema_id),
                defaults={
                    "length": length,
                    "is_sequence": True,
                    "base_observation_set_id": tcn_5_min_obs_set.id,
                },
            )

        # MACD
        MACD.objects.get_or_create(
            id=uuid.UUID("0d1f9d76-8ff2-4f3c-860b-6edfb611a465"),
            defaults={
                "fast": 26,
                "slow": 100,
                "signal": 20,
                "is_sequence": True,
                "base_observation_set_id": tcn_5_min_obs_set.id,
            },
        )

        # BollingerBands
        BollingerBands.objects.get_or_create(
            id=uuid.UUID("1dd0f0fc-08cb-48b4-89ab-8f1763dfa2bf"),
            defaults={
                "length": 20,
                "std_dev": 2.0,
                "is_sequence": True,
                "base_observation_set_id": tcn_5_min_obs_set.id,
            },
        )

        # SuperTrends
        SuperTrend.objects.get_or_create(
            id=uuid.UUID("82673b2e-3483-475a-83c7-a5fdfc40f4d5"),
            defaults={
                "length": 7,
                "multiplier": 3.0,
                "is_sequence": True,
                "base_observation_set_id": tcn_5_min_obs_set.id,
            },
        )

        # FeatureSet
        feature_set, _ = FeatureSet.objects.get_or_create(
            id=PPO_TCN_FUTURES_SET_ID,
            defaults={
                "name": "tcn 5min observation set",
                "base_observation_set_id": tcn_5_min_obs_set.id,
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



        #============================================== Run Configurations ========================================


        for config_class, config_uuid in CONFIG_UUIDS.items():
            RunConfiguration.objects.get_or_create(
                id=config_uuid,
                defaults={
                    "name": config_class.__name__,
                    "description": config_class.__str__(),
                    "blocking": True
                },
            )

        self.stdout.write(self.style.SUCCESS("RunConfiguration fixtures created successfully!"))