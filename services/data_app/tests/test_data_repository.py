from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from django.test import TestCase
from services.data_app.models import BTCFDUSDData
from services.data_app.repositories.data_repository import DataRepository


class DataRepositoryTests(TestCase):

    def setUp(self):
        self.repo = DataRepository()

    def test_insert_tick_data_creates_record(self):
        tick = BTCFDUSDData(
            timestamp=datetime(2025, 8, 15, tzinfo=timezone.utc),
            price=101.5,
            rsi_5=1.0,
            rsi_7=2.0,
            rsi_12=3.0,
            ema_short=4.0,
            ema_mid=5.0,
            ema_long=6.0,
            ema_xlong=7.0,
            macd_line=8.0,
            macd_hist=9.0,
            macd_signal=10.0,
            bb_upper=11.0,
            bb_middle=12.0,
            bb_lower=13.0
        )

        self.repo.insert_tick_data(tick)

        self.assertEqual(BTCFDUSDData.objects.count(), 1)
        stored = BTCFDUSDData.objects.first()
        self.assertEqual(stored.price, 101.5)

    def test_insert_empty_tick_data_creates_record_with_nulls(self):
        ts = datetime(2025, 8, 15, tzinfo=timezone.utc)

        self.repo.insert_empty_tick_data(ts)

        self.assertEqual(BTCFDUSDData.objects.count(), 1)
        stored = BTCFDUSDData.objects.first()
        self.assertIsNone(stored.price)
        self.assertEqual(stored.timestamp, ts)

    @patch("services.data_app.repositories.data_repository.connection_is_good", return_value=True)
    @patch("services.data_app.repositories.data_repository.DataRepository.get_tick_data")
    @patch("services.data_app.repositories.data_repository.DataRepository.insert_tick_data")
    def test_single_poll_and_store_with_connection(
        self, mock_insert, mock_get_data, mock_connection
    ):
        fake_tick = MagicMock(spec=BTCFDUSDData)
        fake_tick.timestamp = datetime(2025, 8, 15, tzinfo=timezone.utc)
        mock_get_data.return_value = fake_tick

        self.repo.single_poll_and_store()

        mock_get_data.assert_called_once()
        mock_insert.assert_called_once_with(fake_tick)

    @patch("services.data_app.repositories.data_repository.connection_is_good", return_value=False)
    @patch("services.data_app.repositories.data_repository.DataRepository.insert_empty_tick_data")
    def test_single_poll_and_store_without_connection(
        self, mock_insert_empty, mock_connection
    ):
        self.repo.single_poll_and_store()
        mock_insert_empty.assert_called_once()
