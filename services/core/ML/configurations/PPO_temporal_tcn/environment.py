from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import runtime_settings
from services.core.models import DerivedfeatureSetMapping, FeatureSet, TickData
from services.core.services.feature_service import DerivedFeatureMethods
from services.types import Actions

WINDOW_TICKS  = getattr(runtime_settings, "DATA_TICKS_WINDOW_LENGTH", 150)
TCN_HIDDEN    = getattr(runtime_settings, "RL_TCN_HIDDEN", 64)
POLICY_PATH   = getattr(runtime_settings, "RL_POLICY_PATH_TCN", None)
TCN_DILATIONS = tuple(getattr(runtime_settings, "RL_TCN_DILATIONS", (1, 2, 4, 8)))

# --- TCN encoder over time-series only ---
class TCNExtractor(nn.Module):
    def __init__(self, in_feats: int, hidden: int = TCN_HIDDEN, dilations=TCN_DILATIONS):
        super().__init__()
        layers = []
        c_in = in_feats
        for d in dilations:
            layers += [nn.Conv1d(c_in, hidden, kernel_size=3, dilation=d, padding=d*2), nn.ReLU()]
            c_in = hidden
        layers += [nn.AdaptiveAvgPool1d(1)]  # global pool over time -> [B, H, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F_ts] -> [B, F_ts, T]
        x = x.transpose(1, 2)
        return self.net(x).squeeze(-1)  # [B, H]


# --- Policy: TCN(time-series) + separate state vector -> actor/critic ---
class TCNActorCriticWithState(nn.Module):
    """
    Forward expects:
      x_ts:    [B, T, F_ts]   (time-series features only)
      x_state: [B, D]         (present-time Markov/state features only)
    """
    def __init__(self, in_feats_ts: int, state_dim: int, hidden: int = TCN_HIDDEN, action_dim: int = 3):
        super().__init__()
        self.tcn = TCNExtractor(in_feats_ts, hidden=hidden)
        self.actor = nn.Sequential(
            nn.Linear(hidden + state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden + state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x_ts: torch.Tensor, x_state: torch.Tensor):
        emb = self.tcn(x_ts)                    # [B, H]
        z = torch.cat([emb, x_state], dim=-1)   # [B, H + D]
        logits = self.actor(z)                  # [B, A]
        value  = self.critic(z)                 # [B, 1]
        return logits, value


# --- Environment (two-input design) ---
class Environment:
    """
    PPO env using a [T, F_ts] time-series observation for the TCN,
    plus a separate [D] present-time state vector (no mirroring into history).
    API: reset(), normalized_reset(), get_inference_action_probs(), step()
    Extra: get_observation_pair() -> (X_ts, x_state)
    """


    def __str__():
        return "A PPO with temporal features [T,F_ts] + separate state [D] (TCN encoder)"


    def __init__(self, feature_set: FeatureSet, tick_df: pd.DataFrame = None,
                 tick_list: List[TickData] = None, initial_cash=1_000_000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.feature_set = feature_set
        self.window_size = WINDOW_TICKS

        if tick_list:
            tick_df = TickData.list_to_env_dataframe(tick_list=tick_list)
        self.data = tick_df
        self.current_step = 0
        self.position: int = 0      # 0=flat, 1=long  (extend later for short)
        self.entry_price: Optional[float] = None

        self.device = torch.device("cpu")
        # lazy policy init when we know dims
        self.policy: Optional[TCNActorCriticWithState] = None
        self._F_ts: Optional[int] = None
        self._D: Optional[int] = None
        self.action_dim = 3  # SELL, HOLD, BUY


    # -------- public API --------
    def reset(self):
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = None
        self.cash = self.initial_cash
        return self._get_observation_ts_only()


    def normalized_reset(self):
        return self.reset()


    def get_inference_action_probs(self):
        self.current_step = getattr(self.feature_set, "window_length", self.window_size)
        X_ts, x_state = self.get_observation_pair()  # [T,F_ts], [D]

        x_ts_t    = torch.tensor(X_ts,    dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, T, F_ts]
        x_state_t = torch.tensor(x_state, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, D]

        # Lazy-build policy with known dims
        if self.policy is None:
            self._F_ts = x_ts_t.shape[-1]
            self._D    = x_state_t.shape[-1]
            self.policy = TCNActorCriticWithState(in_feats_ts=self._F_ts, state_dim=self._D).to(self.device)
            if POLICY_PATH:
                try:
                    sd = torch.load(POLICY_PATH, map_location=self.device)
                    self.policy.load_state_dict(sd, strict=False)
                except Exception:
                    pass
            self.policy.eval()

        with torch.no_grad():
            logits, _ = self.policy(x_ts_t, x_state_t)
            dist = torch.distributions.Categorical(logits=logits)
            p = dist.probs.squeeze(0).cpu().numpy()
            return {Actions.SELL: float(p[0]), Actions.HOLD: float(p[1]), Actions.BUY: float(p[2])}


    # -------- observations --------
    def get_observation_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          X_ts    : [T, F_ts] time-series features only (normalized per feature over window)
          x_state : [D]       present-time Markov/state features only (standardized within vector)
        """
        obs_df = self.data.iloc[self.current_step - self.window_size : self.current_step].copy()
        obs_df = TickData.remove_non_training_fields_in_df(df=obs_df)
        obs_df = obs_df.drop(columns=[c for c in obs_df.columns if obs_df[c].isna().all()])

        # Build time-series features (no state broadcast)
        X_ts = self._build_temporal_matrix(obs_df)  # [T, F_ts]
        # normalize per feature across the window
        mu = X_ts.mean(axis=0, keepdims=True)
        sd = X_ts.std(axis=0, keepdims=True) + 1e-8
        X_ts = ((X_ts - mu) / sd).astype(np.float32)

        # Build present-time state vector (no history)
        x_state = self._build_markov_state(obs_df).astype(np.float32)  # [D]
        return X_ts, x_state


    # Back-compat helper: returns only the time-series part (what step() will emit)
    def _get_observation_ts_only(self) -> np.ndarray:
        X_ts, _ = self.get_observation_pair()
        return X_ts


    def _build_temporal_matrix(self, df: pd.DataFrame) -> np.ndarray:
        close = df["price"].astype(np.float32).to_numpy()
        # log-returns
        r = np.diff(np.log(close + 1e-12), prepend=np.log(close[0] + 1e-12)).astype(np.float32)

        # rolling std of returns
        def roll_std(x, w):
            if len(x) < 2: return np.zeros_like(x, np.float32)
            out = np.zeros_like(x, np.float32)
            for i in range(len(x)):
                seg = x[max(0, i - w + 1): i + 1]
                out[i] = np.std(seg) if len(seg) > 1 else 0.0
            return out
        rv20, rv60 = roll_std(r, 20), roll_std(r, 60)

        # optional indicators (present if precomputed in df)
        cols = []
        for name in ("rsi", "macd", "macd_signal", "ema_fast", "ema_slow"):
            cols.append(df[name].astype(np.float32).to_numpy() if name in df.columns
                        else np.zeros(len(df), np.float32))

        # stack -> [T, F_ts]
        F_list = [r, rv20, rv60, *cols]
        X = np.column_stack(F_list)

        # enforce exact WINDOW_TICKS rows
        if X.shape[0] > WINDOW_TICKS:
            X = X[-WINDOW_TICKS:, :]
        elif X.shape[0] < WINDOW_TICKS:
            pad = np.zeros((WINDOW_TICKS - X.shape[0], X.shape[1]), dtype=np.float32)
            X = np.vstack([pad, X])
        return X


    def _build_markov_state(self, obs_df: pd.DataFrame) -> np.ndarray:
        """
        Present-time (no history) features: position, entry distance, etc.
        Uses DerivedFeatureMethods mapping where methods beginning with 'position_'
        receive (position, entry_price, price); others may compute from obs_df but we take ONLY their *current* value.
        """
        price_now = float(obs_df["price"].iloc[-1])
        derived: Dict[str, float] = {}
        mappings = list(DerivedfeatureSetMapping.objects.filter(feature_set_id=self.feature_set.id))
        for m in mappings:
            if m.derived_feature.method_name.startswith('position_'):
                v = getattr(DerivedFeatureMethods, m.derived_feature.method_name)(self.position, self.entry_price, price_now)
            else:
                v = getattr(DerivedFeatureMethods, m.derived_feature.method_name)(obs_df)

                # Handle scalars vs pandas/numpy sequences
                if isinstance(v, (pd.Series, np.ndarray, list, tuple)):
                    if len(v) > 0:
                        v = float(v[-1])
                    else:
                        v = 0.0
                else:
                    # Already a scalar
                    v = float(v) if v is not None else 0.0

            derived[m.derived_feature.method_name] = v

        x = np.array(list(derived.values()), dtype=np.float32)
        if x.size == 0:
            x = np.zeros(1, dtype=np.float32)
        # standardize within the vector (simple, robust)
        mu = x.mean()
        sd = x.std() + 1e-8
        x = (x - mu) / sd
        return x


    # -------- environment step (long-only sim) --------
    def step(self, action: int):
        """
        Actions:
            -1 = Sell (close long if any)
             0 = Hold
             1 = Buy  (open long if flat)

        Position:
            0 = flat, 1 = long
        """
        price = float(self.data.iloc[self.current_step]['price'])
        realized = 0.0

        if action == 1 and self.position == 0:  # BUY -> open long
            self.position = 1
            self.entry_price = price
        elif action == -1 and self.position == 1:  # SELL -> close long
            realized = price - float(self.entry_price)
            self.cash += realized
            self.position = 0
            self.entry_price = None

        self.current_step += 1
        done = self.current_step >= len(self.data)

        # For back-compat, we return only the time-series observation here.
        obs_ts = self._get_observation_ts_only() if not done else None

        unreal = (price - float(self.entry_price)) if (self.position == 1 and self.entry_price is not None) else 0.0
        info = {
            "cash": self.cash,
            "position": self.position,
            "entry_price": float(self.entry_price) if self.entry_price else 0.0,
            "price": price,
            "realized_pnl": realized,
            "unrealized_pnl": unreal,
        }
        return obs_ts, realized, done, info
    

    def step_with_shorts(self, action: int):
        """
        Like `step`, but allows symmetric long/short behavior with a single-unit position.
        Actions:
            -1 = Sell
            0 = Hold
            1 = Buy

        Position state after using this API:
            -1 = short, 0 = flat, 1 = long

        Rules:
        - If flat (0):
            BUY (1)  -> open LONG at current price
            SELL (-1)-> open SHORT at current price
        - If long (1):
            SELL (-1)-> close LONG, realize PnL = price - entry
            BUY (1)  -> HOLD (no pyramiding)
        - If short (-1):
            BUY (1)  -> close SHORT, realize PnL = entry - price
            SELL (-1)-> HOLD (no pyramiding)
        - HOLD (0) always keeps the current position.
        """
        price = float(self.data.iloc[self.current_step]['price'])
        realized = 0.0

        if action == 1:  # BUY
            if self.position == 0:
                # open long
                self.position = 1
                self.entry_price = price
            elif self.position == -1:
                # close short
                realized = float(self.entry_price) - price
                self.cash += realized
                self.position = 0
                self.entry_price = None
            # if already long -> hold (no pyramiding)

        elif action == -1:  # SELL
            if self.position == 0:
                # open short
                self.position = -1
                self.entry_price = price
            elif self.position == 1:
                # close long
                realized = price - float(self.entry_price)
                self.cash += realized
                self.position = 0
                self.entry_price = None
            # if already short -> hold (no pyramiding)

        # action == 0 -> hold (do nothing)

        # advance time
        self.current_step += 1
        done = self.current_step >= len(self.data)

        # For back-compat, return only the time-series observation
        obs_ts = self._get_observation_ts_only() if not done else None

        # unrealized PnL based on current position
        if self.position == 1 and self.entry_price is not None:
            unreal = price - float(self.entry_price)
        elif self.position == -1 and self.entry_price is not None:
            unreal = float(self.entry_price) - price
        else:
            unreal = 0.0

        info = {
            "cash": self.cash,
            "position": self.position,  # -1 short, 0 flat, 1 long
            "entry_price": float(self.entry_price) if self.entry_price is not None else 0.0,
            "price": price,
            "realized_pnl": realized,
            "unrealized_pnl": unreal,
        }
        return obs_ts, realized, done, info

