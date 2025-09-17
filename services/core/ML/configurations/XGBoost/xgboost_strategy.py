"""
Self-contained XGBoost trading pipeline for Binance Futures 5s ticks.

This module mirrors your PPO environment concepts but implements a supervised
tri-class (SELL/HOLD/BUY) classifier using XGBoost. It integrates with your
Django models (TickData, DataRun, FeatureSet, DerivedfeatureSetMapping, MLModel,
Hyperparameter, TickProbabilities, TrainingSession) and provides:

1) Feature extraction from rolling windows (5-second ticks) + DerivedFeatureMethods.
2) Time-safe tri-class labeling using a forward horizon and fee/edge buffers.
3) Walk-forward cross-validation training (keeps best fold).
4) Model persistence to disk and metadata to DB.
5) Inference helpers (stateless window, env-aware) and persistence of probs.
6) Minimal backtest engine and JSON artifact per fold (actions, probs, equity, Sharpe).
7) A single-call orchestrator: run_futures_xgb_pipeline(...).

Only dependencies outside this file: your Django models + DerivedFeatureMethods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Protocol, runtime_checkable, TYPE_CHECKING
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import balanced_accuracy_score

from django.conf import settings

import runtime_settings
from services.core.models import (
    TickData,
    TrainingFields,
    DataRun,
    FeatureSet,
    DerivedfeatureSetMapping,
    MLModel,
    Hyperparameter,
    TickProbabilities,
    TrainingSession,
)

from xgboost import XGBClassifier as XGBClassifierType


# ===============================
# Configuration, Types & Labeling
# ===============================

@dataclass
class LabelSettings:
    """Settings that define how we create tri-class labels.

    horizon_ticks: how many ticks in the future we look to compute return
    fee_bps:       estimated total round-trip transaction cost in basis points
    min_edge_bps:  additional buffer to avoid over-trading; creates HOLD zone
    """
    horizon_ticks: int = 6      # ~30s for 5-second bars
    fee_bps: float = 2.0        # maker/taker + slippage estimate
    min_edge_bps: float = 0.0   # optional extra buffer


class ActionIndex:
    """Indices for the multi-class output vector."""
    SELL = 0
    HOLD = 1
    BUY = 2


@dataclass(frozen=True)
class ActionProbabilities:
    sell: float
    hold: float
    buy: float
    suggested_action: Optional[int] = None  # -1, 0, 1


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """Minimal interface the backtester/live env must satisfy for typing."""
    data: pd.DataFrame
    feature_set: FeatureSet
    current_step: int
    window_size: int
    position: int
    entry_price: Optional[float]
    data_run: Optional[DataRun]

    def reset(self) -> Optional[np.ndarray]: ...
    def step_with_shorts(self, action: int) -> Tuple[Optional[np.ndarray], float, bool, dict]: ...


# ===============================
# Feature Extraction
# ===============================

def get_training_field_names() -> List[str]:
    """Return names of numeric training fields defined in TrainingFields."""
    return sorted(list(TrainingFields.get_fields()))


def clean_numeric_window(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only training fields, coerce to numeric, fill small gaps."""
    filtered = TickData.remove_non_training_fields_in_df(df)
    for col in filtered.columns:
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce")
    return filtered.fillna(method="ffill").fillna(method="bfill")


# Summary functions applied to each feature over the window
WINDOW_SUMMARY_FUNCTIONS = {
    "last":   lambda arr: float(arr[-1]),
    "mean":   lambda arr: float(np.mean(arr)),
    "std":    lambda arr: float(np.std(arr)),
    "min":    lambda arr: float(np.min(arr)),
    "max":    lambda arr: float(np.max(arr)),
    "p25":    lambda arr: float(np.percentile(arr, 25)),
    "p50":    lambda arr: float(np.percentile(arr, 50)),
    "p75":    lambda arr: float(np.percentile(arr, 75)),
    "slope":  lambda arr: float((arr[-1] - arr[0]) / max(1, len(arr) - 1)),
}


def summarize_window_features(window_df: pd.DataFrame) -> Dict[str, float]:
    """Summarize each feature in the window using a small set of statistics.

    Adds a couple of return-volatility features from price as convenience.
    """
    features: Dict[str, float] = {}

    for column_name in window_df.columns:
        series_values = window_df[column_name].to_numpy(dtype=np.float32)
        if series_values.size == 0 or np.all(np.isnan(series_values)):
            for stat_name in WINDOW_SUMMARY_FUNCTIONS.keys():
                features[f"{column_name}_{stat_name}"] = 0.0
            continue
        for stat_name, summarize in WINDOW_SUMMARY_FUNCTIONS.items():
            try:
                features[f"{column_name}_{stat_name}"] = summarize(series_values)
            except Exception:
                features[f"{column_name}_{stat_name}"] = 0.0

    # Convenience returns/realized-volatilities from price
    if "price" in window_df.columns:
        prices = window_df["price"].to_numpy(np.float32)
        log_returns = np.diff(np.log(prices + 1e-12), prepend=np.log(prices[0] + 1e-12))

        def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
            if arr.size == 0:
                return np.zeros_like(arr, dtype=np.float32)
            out = np.zeros_like(arr, dtype=np.float32)
            for i in range(arr.size):
                segment = arr[max(0, i - window + 1): i + 1]
                out[i] = np.std(segment) if segment.size > 1 else 0.0
            return out

        features["logret_last"]  = float(log_returns[-1]) if log_returns.size else 0.0
        features["rv20_last"]     = float(rolling_std(log_returns, 20)[-1]) if log_returns.size else 0.0
        features["rv60_last"]     = float(rolling_std(log_returns, 60)[-1]) if log_returns.size else 0.0

    return features


# ===============================
# Label Generation
# ===============================

def compute_label_for_index(prices: np.ndarray, current_index: int, settings: LabelSettings) -> int:
    """Compute tri-class label at current_index using future horizon.

    Returns one of {ActionIndex.SELL, ActionIndex.HOLD, ActionIndex.BUY}, or -1 if
    there isn't enough future data to label.
    """
    horizon = settings.horizon_ticks
    if current_index + horizon >= prices.size:
        return -1

    start_price = prices[current_index]
    future_price = prices[current_index + horizon]
    future_return_bps = (future_price / (start_price + 1e-12) - 1.0) * 1e4

    threshold_bps = settings.fee_bps + settings.min_edge_bps
    if future_return_bps > threshold_bps:
        print('action: BUY',)
        return ActionIndex.BUY
    if future_return_bps < -threshold_bps:
        print('action: SELL')
        return ActionIndex.SELL
    return ActionIndex.HOLD


# ===============================
# Data Access & Dataset Building
# ===============================

def fetch_ticks_dataframe(data_run: DataRun) -> pd.DataFrame:
    """Load TickData for a DataRun ordered by timestamp as a DataFrame."""
    qs = (
        TickData.objects.filter(data_run=data_run)
        .only("timestamp", *list(TrainingFields.get_fields()))
        .order_by("timestamp")
    )
    if not qs.exists():
        return pd.DataFrame(columns=["timestamp"] + list(TrainingFields.get_fields()))
    df = pd.DataFrame.from_records(qs.values())
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def select_columns_via_feature_set(feature_set: FeatureSet, numeric_df: pd.DataFrame) -> List[str]:
    """Return the list of training columns to use, honoring FeatureSet contents.

    Your DB stores indicators as families in TrainingFields (e.g., rsi_1..4, ema_1..4,
    macd_line/signal/hist, bb_upper/middle/lower, supertrend_line). We map by family
    prefix and include only columns that exist in the current DataFrame.
    """
    available = set(numeric_df.columns) - {"timestamp"}

    families = {
        "price": {"price"} if "price" in available else set(),
        "rsi": {c for c in available if c.startswith("rsi_")},
        "ema": {c for c in available if c.startswith("ema_")},
        "macd": {c for c in available if c.startswith("macd_")},
        "bb": {c for c in available if c.startswith("bb_")},
        "supertrend": {c for c in available if c.startswith("supertrend_")},
        "stops": {c for c in available if c.endswith("_stop_line")},
    }
    selected = set().union(*families.values())
    return sorted(selected)


def compute_derived_features_for_window(
    feature_set: FeatureSet,
    window_df: pd.DataFrame,
    position_value: int = 0,
    entry_price_value: float = 0.0,
) -> Dict[str, float]:
    """Compute DerivedFeatureMethods mapped to the FeatureSet for the current window.

    Position-based features default to flat during offline training unless you
    pass a position context.
    """
    from services.core.services.feature_service import DerivedFeatureMethods as DF

    derived_values: Dict[str, float] = {}
    mappings = list(DerivedfeatureSetMapping.objects.filter(feature_set_id=feature_set.id))

    current_price = float(window_df["price"].iloc[-1]) if "price" in window_df.columns and len(window_df) else 0.0

    for mapping in mappings:
        method_name = mapping.derived_feature.method_name
        try:
            if method_name.startswith("position_"):
                value = getattr(DF, method_name)(position_value, entry_price_value, current_price)
            else:
                value = getattr(DF, method_name)(window_df)

            if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
                value = float(value[-1]) if len(value) > 0 else 0.0
            else:
                value = float(value) if value is not None else 0.0
        except Exception:
            value = 0.0
        derived_values[f"derived_{method_name}"] = value

    return derived_values


def build_supervised_dataset_for_data_run_fs(
    data_run: DataRun,
    feature_set: FeatureSet,
    window_length: int,
    label_settings: LabelSettings,
) -> Tuple[pd.DataFrame, np.ndarray, List[int]]:
    """FeatureSet-aware (X, y) builder that also adds DerivedFeature values."""
    raw_df = fetch_ticks_dataframe(data_run)
    if raw_df.empty:
        return pd.DataFrame(), np.array([], dtype=int), []

    numeric_df = clean_numeric_window(raw_df)
    if "price" not in numeric_df.columns:
        return pd.DataFrame(), np.array([], dtype=int), []

    selected_columns = select_columns_via_feature_set(feature_set, numeric_df)

    prices = numeric_df["price"].to_numpy(np.float64)
    feature_rows: List[Dict[str, float]] = []
    label_rows: List[int] = []
    used_indices: List[int] = []

    start_index = window_length
    stop_index = len(numeric_df) - label_settings.horizon_ticks

    for end_index in range(start_index, stop_index):
        window_slice = numeric_df.iloc[end_index - window_length: end_index][selected_columns]

        summarized_features = summarize_window_features(window_slice)
        derived_values = compute_derived_features_for_window(
            feature_set=feature_set,
            window_df=window_slice,
            position_value=0,
            entry_price_value=0.0,
        )

        feature_row = {**summarized_features, **derived_values}

        label = compute_label_for_index(prices, end_index - 1, label_settings)
        if label == -1:
            continue

        feature_rows.append(feature_row)
        label_rows.append(label)
        used_indices.append(end_index)

    if not feature_rows:
        return pd.DataFrame(), np.array([], dtype=int), []

    features_dataframe = pd.DataFrame(feature_rows).fillna(0.0)
    labels = np.array(label_rows, dtype=int)
    return features_dataframe, labels, used_indices


# ===============================
# Model Wrapper (Train / Predict / Persist)
# ===============================

class XGBMultiClassModel:
    """XGBoost multi-class classifier with walk-forward CV and JSON IO."""

    def __init__(
        self,
        xgb_hyperparams: Optional[dict] = None,
        max_estimators: int = 1000,
        early_stopping_rounds: int = 50,
        time_splits: int = 5,
    ):
        default_params = dict(
            n_estimators=max_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=42,
        )
        if xgb_hyperparams:
            default_params.update(xgb_hyperparams)

        self.params: dict = default_params
        self.early_stopping_rounds: int = early_stopping_rounds
        self.time_splits: int = time_splits
        self.model: Optional[XGBClassifierType] = None
        self.feature_columns: List[str] = []

    def train_with_walk_forward_cv(self, features: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Train with ordered splits; keep the fold with best balanced accuracy."""
        splitter = TimeSeriesSplit(n_splits=self.time_splits)
        best_balanced_accuracy = -np.inf
        best_model: Optional[XGBClassifierType] = None
        cv_metrics: List[Dict[str, float]] = []

        for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(features), start=1):
            candidate = xgb.XGBClassifier(**self.params)
            X_train, X_valid = features.iloc[train_idx], features.iloc[valid_idx]
            y_train, y_valid = labels[train_idx], labels[valid_idx]

            candidate.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )

            y_pred = candidate.predict(X_valid)
            fold_bal_acc = balanced_accuracy_score(y_valid, y_pred)
            cv_metrics.append({"fold": fold_number, "balanced_accuracy": float(fold_bal_acc)})

            if fold_bal_acc > best_balanced_accuracy:
                print(f"New best model at fold {fold_number} with balanced accuracy {fold_bal_acc:.4f}")
                best_balanced_accuracy = fold_bal_acc
                best_model = candidate

        self.model = best_model
        self.feature_columns = list(features.columns)
        return {"cv": cv_metrics, "best_balanced_accuracy": float(best_balanced_accuracy)}

    def predict_class_probabilities_for_row(self, row_df: pd.DataFrame) -> np.ndarray:
        """Return class probabilities [SELL, HOLD, BUY] for a single-row DataFrame."""
        assert self.model is not None, "Model is not trained or loaded."
        return self.model.predict_proba(row_df[self.feature_columns])

    def save_to_json(self, file_path: str) -> None:
        assert self.model is not None, "Model is not trained or loaded."
        self.model.save_model(file_path)

    def load_from_json(self, file_path: str) -> None:
        classifier = xgb.XGBClassifier(**self.params)
        classifier.load_model(file_path)
        self.model = classifier


# ===============================
# Training Orchestration
# ===============================

def train_model_for_training_session_fs(
    training_session_id,
    label_settings: LabelSettings,
    xgb_hyperparams: Optional[dict] = None,
) -> Tuple[MLModel, Dict]:
    """Aggregate multiple DataRuns from a TrainingSession and train a model."""
    session = (
        TrainingSession.objects.select_related("feature_set", "run_configuration")
        .get(id=training_session_id)
    )

    feature_set: FeatureSet = session.feature_set
    window_length = feature_set.window_length

    feature_frames: List[pd.DataFrame] = []
    label_arrays: List[np.ndarray] = []

    for data_run in session.data_runs.all():
        features_df, labels, _ = build_supervised_dataset_for_data_run_fs(
            data_run=data_run,
            feature_set=feature_set,
            window_length=window_length,
            label_settings=label_settings,
        )
        if not features_df.empty:
            feature_frames.append(features_df)
            label_arrays.append(labels)

    if not feature_frames:
        raise ValueError("No data available to train.")

    features_all = pd.concat(feature_frames, axis=0).reset_index(drop=True)
    labels_all = np.concatenate(label_arrays, axis=0)

    model = XGBMultiClassModel(xgb_hyperparams=xgb_hyperparams)
    cross_val_report = model.train_with_walk_forward_cv(features_all, labels_all)

    ml_model_record = MLModel.objects.create(
        feature_set=feature_set, run_configuration=session.run_configuration
    )

    model_directory = getattr(settings, "ML_MODELS_DIR", os.path.join(settings.BASE_DIR, "models"))
    os.makedirs(model_directory, exist_ok=True)
    model_path = os.path.join(model_directory, f"{ml_model_record.id}.json")
    model.save_to_json(model_path)

    Hyperparameter.objects.create(ml_model=ml_model_record, key="model_path", value=model_path)
    Hyperparameter.objects.create(ml_model=ml_model_record, key="label_settings", value=json.dumps(label_settings.__dict__))
    Hyperparameter.objects.create(ml_model=ml_model_record, key="xgb_params", value=json.dumps(model.params))
    Hyperparameter.objects.create(ml_model=ml_model_record, key="feature_columns", value=json.dumps(model.feature_columns))
    Hyperparameter.objects.create(ml_model=ml_model_record, key="cross_val_report", value=json.dumps(cross_val_report))

    return ml_model_record, cross_val_report


# ===============================
# Inference Helpers
# ===============================

def load_trained_model(ml_model: MLModel) -> XGBMultiClassModel:
    """Create a model wrapper and load weights + feature columns from DB metadata."""
    params = json.loads(ml_model.hyperparameters.get(key="xgb_params").value)
    feature_columns = json.loads(ml_model.hyperparameters.get(key="feature_columns").value)
    model_path = ml_model.hyperparameters.get(key="model_path").value

    model = XGBMultiClassModel(xgb_hyperparams=params)
    model.load_from_json(model_path)
    model.feature_columns = feature_columns
    return model


def load_model_by_id(ml_model_id) -> XGBMultiClassModel:
    ml_model_record = MLModel.objects.get(id=ml_model_id)
    return load_trained_model(ml_model_record)


def compute_action_probabilities_for_window(
    feature_set: FeatureSet,
    window_df: pd.DataFrame,
    model: XGBMultiClassModel,
    position_value: int = 0,
    entry_price_value: float = 0.0,
) -> ActionProbabilities:
    """Stateless core: given a window slice + position context → action probabilities."""
    summarized_features = summarize_window_features(window_df)
    derived_values = compute_derived_features_for_window(
        feature_set=feature_set,
        window_df=window_df,
        position_value=position_value,
        entry_price_value=entry_price_value,
    )

    feature_row = {**summarized_features, **derived_values}
    row_df = pd.DataFrame([feature_row]).fillna(0.0)

    class_probabilities = model.predict_class_probabilities_for_row(row_df)[0]
    return ActionProbabilities(
        sell=float(class_probabilities[ActionIndex.SELL]),
        hold=float(class_probabilities[ActionIndex.HOLD]),
        buy=float(class_probabilities[ActionIndex.BUY]),
    )


def compute_action_probabilities_for_env_step(
    env: EnvironmentProtocol,
    model: XGBMultiClassModel,
    window_length: int,
) -> ActionProbabilities:
    """Pull data/state from env and call the stateless core."""
    assert hasattr(env, "feature_set"), "env must expose feature_set"

    idx = max(env.current_step, window_length)

    df = clean_numeric_window(env.data.copy())
    cols = [c for c in df.columns if c != "timestamp"]
    window_df = df.iloc[idx - window_length: idx][cols]

    pos = int(getattr(env, "position", 0) or 0)
    ep = getattr(env, "entry_price", None)
    epf = float(ep) if ep is not None else 0.0

    return compute_action_probabilities_for_window(
        feature_set=env.feature_set,
        window_df=window_df,
        model=model,
        position_value=pos,
        entry_price_value=epf,
    )


def choose_action_from_probabilities(
    p_sell: float,
    p_hold: float,
    p_buy: float,
    minimum_confidence: float = 0.40,
) -> int:
    """Map class probabilities → discrete action with a confidence floor.
    Returns: -1 (SELL), 0 (HOLD), 1 (BUY)
    """
    candidates = [(-1, float(p_sell)), (0, float(p_hold)), (1, float(p_buy))]
    best_action, best_prob = max(candidates, key=lambda t: t[1])
    return 0 if best_prob < float(minimum_confidence) else best_action


def score_latest_tick_and_persist(
    ml_model_id,
    data_run_id,
    window_length: Optional[int] = None,
    position_value: int = 0,
    entry_price_value: float = 0.0,
) -> Optional[TickProbabilities]:
    """Score the newest tick and store probabilities, including DerivedFeature values."""
    ml_model_record = MLModel.objects.get(id=ml_model_id)
    data_run = DataRun.objects.get(id=data_run_id)

    feature_set = ml_model_record.feature_set
    window_len = window_length or feature_set.window_length

    model = load_trained_model(ml_model_record)

    raw_df = fetch_ticks_dataframe(data_run)
    if len(raw_df) < window_len:
        return None

    numeric_df = clean_numeric_window(raw_df)
    columns = [c for c in numeric_df.columns if c != "timestamp"]
    end_index = len(numeric_df)
    window_df = numeric_df.iloc[end_index - window_len: end_index][columns]

    summarized_features = summarize_window_features(window_df)
    derived_values = compute_derived_features_for_window(
        feature_set=feature_set,
        window_df=window_df,
        position_value=position_value,
        entry_price_value=entry_price_value,
    )

    feature_row = {**summarized_features, **derived_values}
    row_df = pd.DataFrame([feature_row]).fillna(0.0)

    class_probabilities = model.predict_class_probabilities_for_row(row_df)[0]  # [SELL, HOLD, BUY]

    latest_tick = TickData.objects.filter(data_run=data_run).order_by("-timestamp").first()
    result = TickProbabilities.objects.create(
        tick_data=latest_tick,
        buy_prob=float(class_probabilities[ActionIndex.BUY]),
        hold_prob=float(class_probabilities[ActionIndex.HOLD]),
        sell_prob=float(class_probabilities[ActionIndex.SELL]),
    )
    return result


# ===============================
# Backtesting Utilities & Artifacts
# ===============================

def _annualization_factor_for_5s() -> float:
    steps_per_day = (24 * 60 * 60) // 5
    return float(np.sqrt(252 * steps_per_day))


def _compute_sharpe_from_equity(equity_curve: List[float]) -> float:
    if not equity_curve or len(equity_curve) < 3:
        return 0.0
    eq = np.array(equity_curve, dtype=float)
    rets = np.diff(eq) / (eq[:-1] + 1e-12)
    if rets.std(ddof=1) == 0:
        return 0.0
    return float(rets.mean() / (rets.std(ddof=1) + 1e-12) * _annualization_factor_for_5s())


@dataclass(frozen=True)
class BacktestEpisodeResult:
    fold: int
    train_index_range: Tuple[int, int]
    test_index_range: Tuple[int, int]
    sharpe_ratio: float
    cumulative_pnl: float
    equity_curve: List[float]
    actions: List[int]
    timestamps: List[str]
    prices: List[float]
    prob_sell: List[float]
    prob_hold: List[float]
    prob_buy: List[float]


class SimpleBacktestEngine:
    """Minimal backtester that mirrors env position logic for long/short with fees."""

    def __init__(self, feature_set: FeatureSet, df: pd.DataFrame, window_length: int, fee_bps_roundtrip: float = 4.0, min_confidence: float = 0.40):
        self.feature_set = feature_set
        self.df = clean_numeric_window(df)
        self.window_length = window_length
        self.min_conf = min_confidence
        self.fee_frac_roundtrip = fee_bps_roundtrip / 1e4

        self.position: int = 0   # -1 short, 0 flat, 1 long
        self.entry_price: Optional[float] = None
        self.cash: float = 0.0   # PnL in quote per 1 unit notionally
        self.equity_curve: List[float] = []
        self.actions: List[int] = []
        self.timestamps: List[str] = []
        self.prices: List[float] = []
        self.prob_sell: List[float] = []
        self.prob_hold: List[float] = []
        self.prob_buy: List[float] = []

    def step(self, model: XGBMultiClassModel, index: int) -> None:
        cols = [c for c in self.df.columns if c != "timestamp"]
        window_df = self.df.iloc[index - self.window_length: index][cols]
        probs = compute_action_probabilities_for_window(
            feature_set=self.feature_set,
            window_df=window_df,
            model=model,
            position_value=self.position,
            entry_price_value=float(self.entry_price) if self.entry_price is not None else 0.0,
        )
        # store probabilities per step
        self.prob_sell.append(float(probs.sell))
        self.prob_hold.append(float(probs.hold))
        self.prob_buy.append(float(probs.buy))

        action = choose_action_from_probabilities(probs.sell, probs.hold, probs.buy, self.min_conf)

        price = float(self.df.iloc[index]["price"]) if "price" in self.df.columns else 0.0
        realized = 0.0
        fee_close = price * self.fee_frac_roundtrip / 2.0

        if action == 1:  # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                print(f'Long entry_price: {self.entry_price}')
            elif self.position == -1 and self.entry_price is not None:
                realized = (self.entry_price - price) - fee_close
                self.cash += realized
                self.position = 0
                self.entry_price = None
                print(f'Short exit: {price}, realized: {realized}, total PnL: {self.cash}')

        elif action == -1:  # SELL
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                print(f'Short entry_price: {self.entry_price}')
            elif self.position == 1 and self.entry_price is not None:
                realized = (price - self.entry_price) - fee_close
                self.cash += realized
                self.position = 0
                self.entry_price = None
                print(f'Long exit: {price}, realized: {realized}, total PnL: {self.cash}')

        # mark-to-market equity (1 unit)
        unreal = 0.0
        if self.position == 1 and self.entry_price is not None:
            unreal = price - self.entry_price
        elif self.position == -1 and self.entry_price is not None:
            unreal = self.entry_price - price

        equity = self.cash + unreal
        self.equity_curve.append(equity)
        self.actions.append(action)
        ts = self.df.iloc[index]["timestamp"] if "timestamp" in self.df.columns else None
        self.timestamps.append(str(ts) if ts is not None else str(index))
        self.prices.append(price)


# ===============================
# End-to-end: Futures walk-forward trainer + JSON backtest artifacts
# ===============================

def train_test_walk_forward_on_futures(
    label_settings: LabelSettings,
    xgb_params: Optional[dict] = None,
    n_splits: int = 5,
    min_confidence: float = 0.40,
    fee_bps_roundtrip: float = 4.0,
    artifact_name: str = "futures_walkforward.json",
) -> Tuple[MLModel, Dict]:
    """End-to-end: gather futures data runs, train walk-forward, pick best by test Sharpe, and dump JSON artifact."""
    futures_runs = list(DataRun.objects.filter(is_futures=True).order_by("id"))
    if not futures_runs:
        raise ValueError("No futures DataRuns found (is_futures=True).")

    ts = TrainingSession.objects.filter(data_runs__in=futures_runs).select_related("feature_set", "run_configuration").first()
    if not ts:
        raise ValueError("No TrainingSession found that references futures DataRuns; cannot infer FeatureSet.")
    feature_set = ts.feature_set
    window_length = feature_set.window_length

    # Build full dataset across runs
    all_feature_frames: List[pd.DataFrame] = []
    all_label_arrays: List[np.ndarray] = []
    all_raw_frames: List[pd.DataFrame] = []

    for dr in futures_runs[:-1]:
        raw_df = fetch_ticks_dataframe(dr)
        if raw_df.empty:
            continue
        all_raw_frames.append(raw_df.assign(_data_run=str(dr.id)))
        features_df, labels, _ = build_supervised_dataset_for_data_run_fs(
            data_run=dr,
            feature_set=feature_set,
            window_length=window_length,
            label_settings=label_settings,
        )
        if not features_df.empty:
            all_feature_frames.append(features_df)
            all_label_arrays.append(labels)

    if not all_feature_frames:
        raise ValueError("No features built from futures runs.")

    features_all = pd.concat(all_feature_frames, axis=0).reset_index(drop=True)
    labels_all = np.concatenate(all_label_arrays, axis=0)
    raw_all = pd.concat(all_raw_frames, axis=0).reset_index(drop=True)

    splitter = TimeSeriesSplit(n_splits=n_splits)

    best_model: Optional[XGBClassifierType] = None
    best_model_params: Optional[dict] = None
    best_test_sharpe = -np.inf

    episodes: List[BacktestEpisodeResult] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(features_all), start=1):
        model = XGBMultiClassModel(xgb_hyperparams=xgb_params, time_splits=3)
        model.train_with_walk_forward_cv(features_all.iloc[train_idx], labels_all[train_idx])

        test_start = int(test_idx.min())
        test_end_exclusive = int(test_idx.max()) + 1

        bt_start = max(0, test_start - window_length)
        backtest_df = raw_all.iloc[bt_start:test_end_exclusive].copy().reset_index(drop=True)

        engine = SimpleBacktestEngine(
            feature_set=feature_set,
            df=backtest_df,
            window_length=window_length,
            fee_bps_roundtrip=fee_bps_roundtrip,
            min_confidence=min_confidence,
        )

        start_index_local = window_length
        end_index_local = len(backtest_df)
        for idx in range(start_index_local, end_index_local):
            engine.step(model, idx)

        sharpe = _compute_sharpe_from_equity(engine.equity_curve)
        cumulative_pnl = float(engine.equity_curve[-1]) if engine.equity_curve else 0.0

        episode = BacktestEpisodeResult(
            fold=fold,
            train_index_range=(int(train_idx.min()), int(train_idx.max())),
            test_index_range=(int(test_idx.min()), int(test_idx.max())),
            sharpe_ratio=sharpe,
            cumulative_pnl=cumulative_pnl,
            equity_curve=[float(x) for x in engine.equity_curve],
            actions=[int(a) for a in engine.actions],
            timestamps=engine.timestamps,
            prices=[float(p) for p in engine.prices],
            prob_sell=[float(x) for x in engine.prob_sell],
            prob_hold=[float(x) for x in engine.prob_hold],
            prob_buy=[float(x) for x in engine.prob_buy],
        )
        episodes.append(episode)
        print(f"Fold {fold} test Sharpe: {sharpe:.4f}, cumulative PnL: {cumulative_pnl:.4f}")

        if sharpe > best_test_sharpe:
            best_test_sharpe = sharpe
            best_model = model.model
            best_model_params = model.params.copy()

    # Persist best model to MLModel and dump JSON artifact
    ml_model_record = MLModel.objects.create(feature_set=feature_set, run_configuration=ts.run_configuration)

    model_directory = os.path.join(runtime_settings.MODELS_PATH, 'xgboost', ml_model_record.id) # runtime_settings.MODELS_PATH + f"/xgboost/{ml_model_record.id}"
    os.makedirs(model_directory, exist_ok=True)
    model_path = os.path.join(model_directory, "model.json")

    if best_model is None:
        raise RuntimeError("Training produced no valid model.")

    best_wrapper = XGBMultiClassModel(xgb_hyperparams=best_model_params)
    best_wrapper.model = best_model
    best_wrapper.feature_columns = list(features_all.columns)
    best_wrapper.save_to_json(model_path)

    # Save hyperparams/meta
    Hyperparameter.objects.create(ml_model=ml_model_record, key="model_path", value=model_path)
    Hyperparameter.objects.create(ml_model=ml_model_record, key="label_settings", value=json.dumps(label_settings.__dict__))
    Hyperparameter.objects.create(ml_model=ml_model_record, key="xgb_params", value=json.dumps(best_wrapper.params))
    # Hyperparameter.objects.create(ml_model=ml_model_record, key="feature_columns", value=json.dumps(best_wrapper.feature_columns))
    Hyperparameter.objects.create(ml_model=ml_model_record, key="selection_metric", value="test_sharpe")
    Hyperparameter.objects.create(ml_model=ml_model_record, key="selection_value", value=str(best_test_sharpe))

    # JSON artifact
    artifact_path = model_directory + '/training_artifacts.json'

    artifact = {
        "meta": {
            "feature_set_id": str(feature_set.id),
            "window_length": window_length,
            "label_settings": label_settings.__dict__,
            "xgb_params": best_wrapper.params,
            "selection_metric": "test_sharpe",
            "selection_value": best_test_sharpe,
            "model_id": str(ml_model_record.id),
            "model_path": model_path,
        },
        "episodes": [
            {
                "fold": e.fold,
                "train_index_range": e.train_index_range,
                "test_index_range": e.test_index_range,
                "sharpe_ratio": e.sharpe_ratio,
                "cumulative_pnl": e.cumulative_pnl,
                "equity_curve": e.equity_curve,
                "actions": e.actions,
                "timestamps": e.timestamps,
                "prices": e.prices,
                "prob_sell": e.prob_sell,
                "prob_hold": e.prob_hold,
                "prob_buy": e.prob_buy,
            }
            for e in episodes
        ],
    }

    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f)

    Hyperparameter.objects.create(ml_model=ml_model_record, key="artifact_path", value=artifact_path)

    report = {
        "best_test_sharpe": best_test_sharpe,
        "artifact_path": artifact_path,
        "model_id": str(ml_model_record.id),
    }
    return ml_model_record, report


# ===============================
# Simple env backtest & persistence helpers
# ===============================

def backtest_environment_with_model(env: EnvironmentProtocol, model: XGBMultiClassModel, minimum_confidence: float = 0.40) -> pd.DataFrame:
    """Run a simple backtest loop using env.step_with_shorts and the model outputs."""
    _ = env.reset()
    is_done = False
    records: List[Dict] = []

    while not is_done:
        probs = compute_action_probabilities_for_env_step(env, model, window_length=env.window_size)
        action = choose_action_from_probabilities(
            p_sell=probs.sell,
            p_hold=probs.hold,
            p_buy=probs.buy,
            minimum_confidence=minimum_confidence,
        )
        _, realized, is_done, info = env.step_with_shorts(action)
        info["action"] = action
        info["realized"] = realized
        info["prob_sell"] = probs.sell
        info["prob_hold"] = probs.hold
        info["prob_buy"] = probs.buy
        records.append(info)

    return pd.DataFrame(records)


def score_env_step_and_persist(
    env: EnvironmentProtocol,
    ml_model_id,
    minimum_confidence: Optional[float] = None,
) -> Tuple[ActionProbabilities, Optional[TickProbabilities]]:
    """Compute action probabilities for the current env step and optionally persist."""
    model = load_model_by_id(ml_model_id)

    window_len = getattr(env, "window_size", getattr(env, "window_length", None))
    if window_len is None:
        raise ValueError("env must expose window_size/window_length")

    idx = max(env.current_step, window_len)

    df = clean_numeric_window(env.data.copy())
    cols = [c for c in df.columns if c != "timestamp"]
    window_df = df.iloc[idx - window_len: idx][cols]

    pos = int(getattr(env, "position", 0) or 0)
    ep = getattr(env, "entry_price", None)
    epf = float(ep) if ep is not None else 0.0

    probs = compute_action_probabilities_for_window(
        feature_set=env.feature_set,
        window_df=window_df,
        model=model,
        position_value=pos,
        entry_price_value=epf,
    )

    if minimum_confidence is not None:
        action = choose_action_from_probabilities(
            p_sell=probs.sell,
            p_hold=probs.hold,
            p_buy=probs.buy,
            minimum_confidence=minimum_confidence,
        )
        probs = ActionProbabilities(sell=probs.sell, hold=probs.hold, buy=probs.buy, suggested_action=action)

    data_run = getattr(env, "data_run", None)
    persisted = None
    if data_run is not None:
        latest_tick = TickData.objects.filter(data_run=data_run).order_by("-timestamp").first()
        if latest_tick is not None:
            persisted = TickProbabilities.objects.create(
                tick_data=latest_tick,
                buy_prob=float(probs.buy),
                hold_prob=float(probs.hold),
                sell_prob=float(probs.sell),
            )

    return probs, persisted


# ===============================
# Single-call entry point
# ===============================

def run_futures_xgb_pipeline(
    horizon_ticks: int = 6,
    fee_bps: float = 2.0,
    min_edge_bps: float = 1.0,
    xgb_params: Optional[dict] = None,
    n_splits: int = 5,
    min_confidence: float = 0.40,
    fee_bps_roundtrip: float = 4.0,
    artifact_name: str = "futures_walkforward.json",
    env: Optional[EnvironmentProtocol] = None,
    env_results_path: Optional[str] = None,
) -> dict:
    """Single-call entry point.

    1) Trains on all futures DataRuns with walk-forward CV and selects best by test Sharpe.
    2) Dumps JSON artifact with per-step actions + probabilities + equity.
    3) (Optional) Backtests a provided Environment and saves CSV.
    """
    label_settings = LabelSettings(horizon_ticks=horizon_ticks, fee_bps=fee_bps, min_edge_bps=min_edge_bps)

    ml_model, report = train_test_walk_forward_on_futures(
        label_settings=label_settings,
        xgb_params=xgb_params,
        n_splits=n_splits,
        min_confidence=min_confidence,
        fee_bps_roundtrip=fee_bps_roundtrip,
        artifact_name=artifact_name,
    )

    result = {
        "model_id": str(ml_model.id),
        "artifact_path": report.get("artifact_path"),
        "best_test_sharpe": report.get("best_test_sharpe"),
    }

    if env is not None:
        model = load_trained_model(ml_model)
        df_env = backtest_environment_with_model(env, model, minimum_confidence=min_confidence)
        result["env_backtest_rows"] = int(len(df_env))
        if env_results_path:
            try:
                df_env.to_csv(env_results_path, index=False)
                result["env_results_path"] = env_results_path
            except Exception as e:
                result["env_results_save_error"] = str(e)

    return result


__all__ = [
    # Config & types
    "LabelSettings", "ActionIndex", "ActionProbabilities", "EnvironmentProtocol",
    # Building/training
    "build_supervised_dataset_for_data_run_fs",
    "train_model_for_training_session_fs",
    "train_test_walk_forward_on_futures",
    # Model wrapper
    "XGBMultiClassModel",
    # Inference core/adapters
    "compute_action_probabilities_for_window",
    "compute_action_probabilities_for_env_step",
    # Persistence helpers
    "load_trained_model", "load_model_by_id",
    "score_latest_tick_and_persist", "score_env_step_and_persist",
    # Backtest utils
    "choose_action_from_probabilities", "backtest_environment_with_model",
    # Single-call pipeline
    "run_futures_xgb_pipeline",
]
