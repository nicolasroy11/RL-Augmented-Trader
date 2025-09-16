from __future__ import annotations
from collections import namedtuple
import time
from typing import Dict, List, Tuple, Optional, NamedTuple
from uuid import UUID
import pandas as pd
import runtime_settings
from services.core.ML.configurations.fixture_config import DEFAULT_FEATURE_SET_ID, CONFIG_UUIDS
from services.core.ML.helpers import json_log_training_progression
from services.core.models import FeatureSet, Hyperparameter, MLModel, RunConfiguration, TickData, TrainingSession

import random
import torch
import torch.nn as nn
import torch.optim as optim
from django.db.models.query import QuerySet

from services.core.dtos.policy_gradient_results_dto import EpisodeResultsDto, PPOTCNTrainingResults
import numpy as np

# Two-input env & policy (temporal TCN + markov state)
from services.core.ML.configurations.PPO_temporal_tcn.environment import (
    Environment as EnvironmentTCN,
    TCNActorCriticWithState,
)

DATA_TABLE_NAME = TickData._meta.db_table


# ----------------------------
# Advantage estimation (GAE)
# ----------------------------
class EvalMetrics(NamedTuple):
    pnl: float
    sharpe: float
    max_drawdown: float
    trades: int


def compute_gae(
    rewards: torch.Tensor,      # [N]
    values: torch.Tensor,       # [N]
    dones: torch.Tensor,        # [N] in {0.0, 1.0}
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation.
    Returns:
        advantages: [N]
        returns:    [N] = advantages + values
    """
    n_steps = rewards.shape[0]
    advantages = torch.zeros(n_steps, dtype=torch.float32, device=rewards.device)
    gae = 0.0
    for t in reversed(range(n_steps)):
        next_value = values[t + 1] if (t + 1) < n_steps else torch.tensor(0.0, device=values.device)
        not_done = 1.0 - (dones[t + 1] if (t + 1) < n_steps else 1.0)
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


class RLRepository:
    """
    PPO trainer with per-run chronological train/test splits:
      - For each data_run_id: split first (1 - test_ratio) into TRAIN, last test_ratio into TEST (validation).
      - Best checkpoint chosen by validation Sharpe with drawdown cap.
      - Final evaluation on the aggregate TEST set (all per-run hold-outs).
      - Uses HOLD steps (action=0) to advance non-decision ticks (no index jumping).
    """

    # -------------------- Chunking by time gaps --------------------
    def get_valid_data_chunks(
        self,
        queryset: QuerySet[TickData],
        window_size: int,
        min_windows: int = 10
    ) -> Dict[UUID, pd.DataFrame]:
        """
        Separate ticks into contiguous DataFrames (regimes), splitting when gap > 1 minute.
        Keeps chunks with at least `window_size * min_windows` rows.
        Returns a dict keyed by data_run_id -> chunk dataframe.
        """
        df: pd.DataFrame = pd.DataFrame.from_records(queryset.values()).sort_values('timestamp').reset_index(drop=True)
        if df.empty:
            return {}

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['gap'] = df['timestamp'].diff().fillna(pd.Timedelta(seconds=0))
        split_indices: List[int] = df.index[df['gap'] > pd.Timedelta(minutes=1)].tolist()
        split_indices = [0] + split_indices + [len(df)]

        chunks: Dict[UUID, pd.DataFrame] = {}
        for start, end in zip(split_indices[:-1], split_indices[1:]):
            chunk_df: pd.DataFrame = df.iloc[start:end].copy()
            chunk_df.drop(columns=['gap'], inplace=True)
            if len(chunk_df) >= window_size * min_windows:
                run_id = chunk_df.iloc[0]['data_run_id']
                # If multiple regimes belong to the same run_id, append them
                if run_id in chunks:
                    # concatenate, preserving order
                    chunks[run_id] = pd.concat([chunks[run_id], chunk_df], ignore_index=True)
                else:
                    chunks[run_id] = chunk_df.reset_index(drop=True)
        return chunks

    # -------------------- Per-run chronological split --------------------
    def split_train_val_per_run(
        self,
        chunks: Dict[UUID, pd.DataFrame],
        window_size: int,
        test_ratio: float = 0.2,
        min_windows: int = 10,
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Chronologically split each run into train/test by `test_ratio`.
        Ensures both splits are large enough (>= window_size * min_windows).
        Aggregates across runs and returns:
            (all_train_chunks, all_val_chunks)
        """
        min_len = window_size * min_windows
        all_train: List[pd.DataFrame] = []
        all_val: List[pd.DataFrame] = []

        for run_id, df in chunks.items():
            n = len(df)
            if n < 2 * min_len:
                # too small to split reliably; skip this run
                continue

            split_idx = int(n * (1.0 - test_ratio))
            split_idx = max(min_len, min(split_idx, n - min_len))  # keep space for rolling windows on both sides

            train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
            val_df   = df.iloc[split_idx:].copy().reset_index(drop=True)

            if len(train_df) >= min_len:
                all_train.append(train_df)
            if len(val_df) >= min_len:
                all_val.append(val_df)

        if not all_train:
            raise ValueError("No per-run train splits met minimum length.")
        if not all_val:
            raise ValueError("No per-run validation splits met minimum length.")

        return all_train, all_val

    # -------------------- Utility: BH PnL (used only for reporting) --------------------
    def _buy_and_hold_pnl_for_df(self, df: pd.DataFrame) -> float:
        if df is None or len(df) < 2:
            return 0.0
        first_price = float(df["price"].iloc[0])
        last_price = float(df["price"].iloc[-1])
        return last_price - first_price

    # -------------------- Greedy evaluation on chunks --------------------
    @torch.no_grad()
    def _evaluate_policy_on_chunks(
        self,
        policy: TCNActorCriticWithState,
        chunks: List[pd.DataFrame],
        feature_set: FeatureSet,
        decision_every_k: int,
    ) -> EvalMetrics:
        """
        Greedy (argmax) evaluation with shorts allowed (uses env.step_with_shorts).
        Counts trades ONLY when a round-trip is closed: info['realized_pnl'] != 0.
        """
        device = torch.device("cpu")
        policy.eval()

        cumulative_pnl = 0.0
        equity_curve: List[float] = []
        trades_count = 0

        for chunk in chunks:
            env = EnvironmentTCN(tick_df=chunk, feature_set=feature_set)
            env.normalized_reset()

            if len(env.data) == 0:
                continue

            done = False
            step = 0
            X_ts, x_state = env.get_observation_pair()

            while not done:
                step += 1
                if (step % decision_every_k) != 0:
                    # advance time through env (HOLD), not by index jump
                    _, _, done, _info = env.step_with_shorts(0)
                    equity_curve.append(cumulative_pnl)
                    continue

                if env.current_step >= len(env.data):
                    break

                ts_tensor = torch.tensor(X_ts, dtype=torch.float32, device=device).unsqueeze(0)
                state_tensor = torch.tensor(x_state, dtype=torch.float32, device=device).unsqueeze(0)

                logits, _ = policy(ts_tensor, state_tensor)
                a_idx = int(torch.argmax(logits, dim=-1).item())
                env_action = (-1 if a_idx == 0 else (0 if a_idx == 1 else 1))

                _, reward, done, info = env.step_with_shorts(env_action)

                if (env_action !=0): print(info)
                cumulative_pnl += reward
                equity_curve.append(cumulative_pnl)

                # count completed round-trips only
                if abs(float(info.get("realized_pnl", 0.0))) > 0.0:
                    trades_count += 1

                if not done:
                    X_ts, x_state = env.get_observation_pair()

            # (Optional) one final fair decision at terminal if still in position — omitted for purity

        pnls = np.array(equity_curve, dtype=float)
        if pnls.size == 0:
            return EvalMetrics(pnl=0.0, sharpe=0.0, max_drawdown=0.0, trades=0)

        running_max = np.maximum.accumulate(pnls)
        drawdowns = running_max - pnls
        max_drawdown = float(drawdowns.max()) if drawdowns.size else 0.0
        step_returns = np.diff(pnls) if pnls.size > 1 else np.array([0.0])
        sharpe = float((step_returns.mean() / (step_returns.std() + 1e-9)) * np.sqrt(252)) if step_returns.size else 0.0

        return EvalMetrics(
            pnl=float(pnls[-1]),
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            trades=int(trades_count),
        )

    # -------------------- Training entrypoint --------------------
    def run_ppo(
        self,
        window_size=10,
        num_episodes=250,
        gamma=0.95,
        gae_lambda=0.95,
        lr=1e-3,
        clip_epsilon=0.4,
        ppo_epochs=8,
        batch_size=32,
        ent_coef=0.02,
        decision_every_k=1,
        feature_set_id: UUID = None,
        is_futures=True
    ) -> TCNActorCriticWithState:
        # Session & configs
        session = TrainingSession()
        session.num_episodes = num_episodes

        run_config = RunConfiguration.objects.get(id=CONFIG_UUIDS[EnvironmentTCN])
        session.run_configuration = run_config

        if not feature_set_id:
            feature_set_id = DEFAULT_FEATURE_SET_ID
        session.feature_set = FeatureSet.objects.get(id=feature_set_id)

        # Data -> contiguous per-run chunks (regimes)
        queryset = TickData.objects.order_by('timestamp').filter(data_run__is_futures=is_futures).all()
        self.all_chunks = self.get_valid_data_chunks(queryset, window_size=window_size, min_windows=10)
        if not self.all_chunks:
            raise ValueError("No valid chunks found with enough rows.")

        # Per-run chronological split (no leakage)
        per_run_test_ratio = float(getattr(runtime_settings, "PER_RUN_TEST_RATIO", 0.2))
        self.train_chunks, self.val_chunks = self.split_train_val_per_run(
            self.all_chunks, test_ratio=per_run_test_ratio, window_size=window_size, min_windows=10
        )
        print(f"[SPLIT] train_chunks={len(self.train_chunks)}, val_chunks={len(self.val_chunks)}, "
              f"test_ratio={per_run_test_ratio}")

        # Save session metadata (keep only training run_ids for traceability)
        session.save()
        session.data_runs.set(list(self.all_chunks.keys()))
        self.session = session

        # Model record
        model = MLModel()
        model.feature_set = session.feature_set
        model.run_configuration = session.run_configuration
        model.save()
        self.model = model

        # Read & log hyperparameters
        val_max_dd_cap = float(getattr(runtime_settings, "VAL_MAX_DRAWDOWN_CAP", 0.15))  # robust selection
        decision_k = decision_every_k
        for k, v in {
            "lr": lr,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_epsilon": clip_epsilon,
            "ppo_epochs": ppo_epochs,
            "batch_size": batch_size,
            "ent_coef": ent_coef,
            "per_run_test_ratio": per_run_test_ratio,
            "val_max_drawdown_cap": val_max_dd_cap,
            "decision_every_k": decision_k,
        }.items():
            Hyperparameter(ml_model=model, key=k, value=v).save()

        # Lazy-build policy after first observation pair
        self.policy: Optional[TCNActorCriticWithState] = None
        self.optimizer = None

        # Track best validation Sharpe
        self.best_val_sharpe: float = -1e9
        self.best_model_path: Optional[str] = None
        self.val_max_dd_cap = val_max_dd_cap
        self.decision_every_k = decision_k

        # Train
        results = self._run_ppo(
            num_episodes, gamma, gae_lambda, clip_epsilon, ppo_epochs, batch_size, ent_coef, lr, decision_every_k
        )

        # --- Final evaluation on the aggregate per-run TEST set ---
        # policy_for_test = self.policy  # evaluate the last policy by default
        # Or evaluate best checkpoint instead (uncomment):
        if self.best_model_path:
            self.policy.load_state_dict(torch.load(self.best_model_path, map_location="cpu"))
            policy_for_test = self.policy

        final_val_metrics = self._evaluate_policy_on_chunks(
            policy_for_test,
            self.val_chunks,
            self.session.feature_set,
            decision_every_k=self.decision_every_k,
        )
        print("[FINAL VALID] "
              f"PnL={final_val_metrics.pnl:.2f}, "
              f"Sharpe={final_val_metrics.sharpe:.2f}, "
              f"MaxDD={final_val_metrics.max_drawdown:.3f}, "
              f"Trades={final_val_metrics.trades}")

        # Populate typed results
        results.holdout_pnl = final_val_metrics.pnl
        results.holdout_sharpe = final_val_metrics.sharpe
        results.holdout_max_drawdown = final_val_metrics.max_drawdown
        results.holdout_trades = final_val_metrics.trades
        results.best_model_path = self.best_model_path
        return results

    # -------------------- Training loop --------------------
    def _run_ppo(
        self,
        num_episodes,
        gamma,
        gae_lambda,
        clip_epsilon,
        ppo_epochs,
        batch_size,
        ent_coef,
        lr,
        decision_every_k
    ) -> PPOTCNTrainingResults:

        all_episode_metrics: List[EpisodeResultsDto] = []
        device = torch.device("cpu")

        for episode_number in range(num_episodes):
            print(f"\n=== Starting Episode {episode_number} ===")

            # Buffers for rollout
            ts_buffer: List[torch.Tensor] = []
            state_buffer: List[torch.Tensor] = []
            action_buffer: List[torch.Tensor] = []
            logprob_buffer: List[torch.Tensor] = []
            value_buffer: List[torch.Tensor] = []
            reward_buffer: List[torch.Tensor] = []
            done_buffer: List[torch.Tensor] = []
            action_probs_log: List[List[float]] = []

            cumulative_pnl = 0.0
            running_episode_pnls: List[float] = []
            prices_logged: List[float] = []

            # Shuffle training chunks each episode
            train_chunk_list = list(self.train_chunks)
            random.shuffle(train_chunk_list)

            # ===== Rollout on TRAIN chunks =====
            for chunk in train_chunk_list:
                step = 0
                prices_logged.extend(list(chunk['price']))
                env = EnvironmentTCN(tick_df=chunk, feature_set=self.session.feature_set)
                env.normalized_reset()

                # First observation pair
                X_ts, x_state = env.get_observation_pair()

                # Lazy policy build
                if self.policy is None:
                    T, F_ts = X_ts.shape
                    D = x_state.shape[-1]
                    self.policy = TCNActorCriticWithState(in_feats_ts=F_ts, state_dim=D, action_dim=3).to(device)
                    self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

                done = False
                while not done:
                    step += 1

                    # Advance non-decision ticks via HOLD step (no index jumping)
                    if (step % self.decision_every_k) != 0:
                        _, _, done, _info = env.step_with_shorts(0)
                        # No storage on non-decision steps; we log only decision transitions
                        # But keep equity curve for metrics
                        running_episode_pnls.append(cumulative_pnl)
                        continue

                    if env.current_step >= len(env.data):
                        done = True
                        continue

                    ts_t = torch.tensor(X_ts, dtype=torch.float32, device=device).unsqueeze(0)    # [1, T, F_ts]
                    st_t = torch.tensor(x_state, dtype=torch.float32, device=device).unsqueeze(0) # [1, D]

                    logits, value = self.policy(ts_t, st_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                    a_idx = int(action.item())
                    env_action = (-1 if a_idx == 0 else (0 if a_idx == 1 else 1))

                    _, reward, done, info = env.step_with_shorts(env_action)

                    # Store transitions at decision steps
                    ts_buffer.append(ts_t.squeeze(0).detach())
                    state_buffer.append(st_t.squeeze(0).detach())
                    action_buffer.append(action.detach())
                    logprob_buffer.append(log_prob.detach())
                    value_buffer.append(value.detach().squeeze(-1))
                    reward_buffer.append(torch.tensor([reward], dtype=torch.float32, device=device))
                    done_buffer.append(torch.tensor([1.0 if done else 0.0], dtype=torch.float32, device=device))
                    action_probs_log.append(list(dist.probs.detach().cpu().numpy().flatten()))

                    cumulative_pnl += reward
                    running_episode_pnls.append(cumulative_pnl)

                    # Next observation
                    if not done:
                        X_ts, x_state = env.get_observation_pair()

                    # (Optional) debug print
                    print("step:", f'{step}/{episode_number}', "probs:", dist.probs.detach().cpu().numpy(), "cum_pnl:", cumulative_pnl)

            # --- Stack tensors for PPO ---
            if not reward_buffer:
                # No decision steps collected; skip updates for this episode
                print("No decision steps collected; skipping PPO update.")
                continue

            rewards = torch.cat(reward_buffer).squeeze(-1)
            values = torch.cat(value_buffer).squeeze(-1)
            log_probs_old = torch.cat(logprob_buffer)
            ts_tensor = torch.stack(ts_buffer, dim=0)
            state_tensor = torch.stack(state_buffer, dim=0)
            actions_tensor = torch.cat(action_buffer)
            dones = torch.cat(done_buffer).squeeze(-1)

            # --- GAE + normalize ---
            advantages, returns = compute_gae(rewards, values, dones, gamma=gamma, lam=gae_lambda)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # --- PPO updates ---
            num_steps = ts_tensor.shape[0]
            for _ in range(ppo_epochs):
                for i in range(0, num_steps, batch_size):
                    j = slice(i, i + batch_size)
                    batch_ts = ts_tensor[j]
                    batch_state = state_tensor[j]
                    batch_actions = actions_tensor[j]
                    batch_returns = returns[j]
                    batch_advantages = advantages[j]
                    batch_log_probs_old = log_probs_old[j]

                    logits, value_pred = self.policy(batch_ts, batch_state)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_probs_new = dist.log_prob(batch_actions)

                    ratio = (log_probs_new - batch_log_probs_old).exp()
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(value_pred.squeeze(-1), batch_returns)
                    entropy = dist.entropy().mean()

                    loss = policy_loss + 0.5 * value_loss - ent_coef * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()

            # --- Episode training metrics ---
            episode_pnls_array = np.array(running_episode_pnls)
            running_max = np.maximum.accumulate(episode_pnls_array) if episode_pnls_array.size else np.array([])
            drawdowns = (running_max - episode_pnls_array) if episode_pnls_array.size else np.array([])
            max_drawdown = float(drawdowns.max()) if drawdowns.size else 0.0
            final_pnl = float(episode_pnls_array[-1]) if episode_pnls_array.size else 0.0
            buy_and_hold_pnl = sum(
                chunk.iloc[-1]['price'] - chunk.iloc[0]['price'] for chunk in train_chunk_list
            ) if train_chunk_list else 0.0
            step_returns = np.diff(episode_pnls_array) if episode_pnls_array.size else np.array([])
            sharpe_ratio = (step_returns.mean() / (step_returns.std() + 1e-9) * np.sqrt(252)) if step_returns.size else 0.0

            # --- Validation (greedy) on aggregate per-run TEST chunks ---
            # val_metrics = self._evaluate_policy_on_chunks(
            #     self.policy, self.val_chunks, self.session.feature_set, decision_every_k=self.decision_every_k
            # )

            # Save every episode
            model_path = f"pth_files/{self.model.id}_tcn2in_gae_{episode_number}.pth"
            torch.save(self.policy.state_dict(), model_path)
            print(f"Saved policy to {model_path}")

            val_metrics, val_csv = evaluate_policy_on_chunks_with_log(
                policy=self.policy,
                chunks=list(self.val_chunks),
                feature_set=self.session.feature_set,
                episode_number=episode_number,
                decision_every_k=decision_every_k,
                csv_dir="logs",
                csv_prefix="val_trades",
                log_every_hold=50,  # set 1 if you want every HOLD printed
            )

            # Logs
            print(
                f"Episode {episode_number} TRAIN  → PnL {final_pnl:.2f}, "
                f"BH {buy_and_hold_pnl:.2f}, MaxDD {max_drawdown:.3f}, Sharpe {sharpe_ratio:.2f}"
            )

            print(f"Episode {episode_number} VALID → "
                f"PnL {val_metrics.pnl:.2f}, "
                f"MaxDD {val_metrics.max_drawdown:.3f}, "
                f"Sharpe {val_metrics.sharpe:.2f}, "
                f"Trades {val_metrics.trades}, "
                f"CSV: {val_csv or '—'}")

            # Save best-so-far by validation Sharpe with DD cap
            if val_metrics.sharpe > getattr(self, "best_val_sharpe", -1e9) and val_metrics.max_drawdown <= self.val_max_dd_cap:
                self.best_val_sharpe = val_metrics.sharpe
                self.best_model_path = f"pth_files/{self.model.id}_BEST.pth"
                torch.save(self.policy.state_dict(), self.best_model_path)
                print(f"[BEST] New best validation Sharpe {val_metrics.sharpe:.3f} "
                      f"(DD {val_metrics.max_drawdown:.3f}, trades {val_metrics.trades}) "
                      f"→ saved {self.best_model_path}")



            # --- Record into typed EpisodeResultsDto (includes validation) ---
            ep = EpisodeResultsDto()
            ep.episode_number = episode_number
            ep.final_pnl = final_pnl
            ep.episode_pnls = episode_pnls_array.tolist()
            ep.running_max = running_max.tolist() if running_max.size else []
            ep.drawdowns = drawdowns.tolist() if drawdowns.size else []
            ep.max_drawdown = max_drawdown
            ep.buy_and_hold_pnl = buy_and_hold_pnl
            ep.sharpe_ratio = sharpe_ratio
            ep.action_probs = action_probs_log
            ep.prices = prices_logged

            # Validation fields
            ep.validation_pnl = val_metrics.pnl
            ep.validation_sharpe = val_metrics.sharpe
            ep.validation_max_drawdown = val_metrics.max_drawdown
            ep.validation_trades = val_metrics.trades
            ep.model_path = model_path

            all_episode_metrics.append(ep)

            # Persist rolling log each episode
            json_log_training_progression(all_episode_metrics, f'PPO_TCN_2in_GAE_{self.session.id}')

        # Build results; final hold-out (aggregate per-run TEST) will be filled in run_ppo()
        results = PPOTCNTrainingResults()
        results.episode_results = all_episode_metrics
        results.holdout_pnl = 0.0
        results.holdout_sharpe = 0.0
        results.holdout_max_drawdown = 0.0
        results.holdout_trades = 0
        results.best_model_path = None
        return results
    

# validation_eval.py


from typing import List, NamedTuple, Optional, Tuple
import os
import numpy as np
import pandas as pd
import torch

import runtime_settings
from services.core.models import FeatureSet
from services.core.ML.configurations.PPO_temporal_tcn.environment import (
    Environment as EnvironmentTCN,
    TCNActorCriticWithState,
)


class EvalMetrics(NamedTuple):
    pnl: float
    sharpe: float
    max_drawdown: float
    trades: int


class ValTrade(NamedTuple):
    run_id: Optional[str]
    episode: int
    step_index: int            # index of decision step within this evaluation
    timestamp: str
    action_idx: int            # 0=sell,1=hold,2=buy (argmax)
    env_action: int            # -1,0,1 actually sent to env
    position_before: int       # -1,0,1
    position_after: int        # -1,0,1
    price: float
    realized_pnl: float        # non-zero only on close
    unrealized_pnl: float



# Named containers for metrics & trade rows
# EvalMetrics = namedtuple("EvalMetrics", ["pnl", "sharpe", "max_drawdown", "trades"])
# ValTrade = namedtuple(
#     "ValTrade",
#     [
#         "run_id", "episode", "step_index", "timestamp",
#         "action_idx", "env_action",
#         "position_before", "position_after",
#         "price", "realized_pnl", "unrealized_pnl",
#     ],
# )

@torch.no_grad()
def evaluate_policy_on_chunks_with_log(
    policy,
    chunks: List[pd.DataFrame],
    feature_set,
    decision_every_k: int,
    episode_number: int = -1,
    csv_dir: str = "logs",
    csv_prefix: str = "val_trades",
    log_every_hold: int = 1,  # print each HOLD; set to 50 to thin it
) -> Tuple[EvalMetrics, str]:
    """
    Validation evaluator with strict entry gating & trailing-stop exits.

    Actions mapping:
      model logits -> probs [sell, hold, buy], indices 0/1/2
      env_action   -> -1 (sell), 0 (hold), 1 (buy)
    """
    device = torch.device("cpu")
    policy.eval()

    # ---- knobs (override in runtime_settings if you prefer) ----
    BUY_OPEN_THRESHOLD  = float(getattr(runtime_settings, "VAL_BUY_OPEN_THRESHOLD", 0.50))
    SELL_OPEN_THRESHOLD = float(getattr(runtime_settings, "VAL_SELL_OPEN_THRESHOLD", 0.50))


    FEE_BPS            = float(getattr(runtime_settings, "SIM_FEE_BPS", 4.0))  # round-trip fee in bps
    TRAIL_ACTIVATE_BPS = float(getattr(runtime_settings, "VAL_TRAIL_ACTIVATE_BPS", 2.0 * FEE_BPS))
    TRAIL_BPS          = float(getattr(runtime_settings, "VAL_TRAIL_BPS", 10.0))

    # If your env step gates NEW opens by an edge threshold, keep it ≥ fees:
    # runtime_settings.SIM_MIN_EDGE_BPS = max( getattr(...), FEE_BPS )

    os.makedirs(csv_dir, exist_ok=True)
    cumulative_pnl = 0.0
    equity_curve: List[float] = []
    trades_count = 0
    trade_log: List[ValTrade] = []
    csv_path = ""

    global_step_index = 0  # increments on each decision step (not every hold)

    for chunk in chunks:
        if chunk is None or len(chunk) == 0:
            continue

        # Per-chunk trailing/cooldown state
        trail_armed = False
        trail_anchor: Optional[float] = None  # peak (long) or trough (short)
        trail_stop: Optional[float] = None

        env = EnvironmentTCN(tick_df=chunk, feature_set=feature_set)
        env.normalized_reset()
        if env.current_step >= len(env.data):
            continue

        # initial observation pair
        X_ts, x_state = env.get_observation_pair()
        done = False
        loop_step = 0

        while not done:
            loop_step += 1

            # --------- Non-decision ticks: HOLD ---------
            if (loop_step % decision_every_k) != 0:
                _, _, done, info = env.step_with_shorts(0)
                if log_every_hold > 0 and (env.current_step % log_every_hold == 0):
                    price_now = float(info.get("price", 0.0))
                    unreal = float(info.get("unrealized_pnl", 0.0))
                    print(
                        f"[VAL] step={env.current_step:5d} price: {price_now:,.2f} "
                        f"aaction: {0:+d} position: {env.position:+d} event=hold "
                        f"real=0.00 unreal={unreal:,.2f}",
                        flush=True,
                    )
                equity_curve.append(cumulative_pnl)
                continue

            # Avoid terminal "decisions" at last row
            if env.current_step >= len(env.data) - 1:
                break

            # ------------------ DECISION SECTION (start) ------------------
            cur_px = float(env.data.iloc[env.current_step]['price'])

            # Forward pass
            ts_t = torch.tensor(X_ts, dtype=torch.float32, device=device).unsqueeze(0)   # [1,T,F]
            st_t = torch.tensor(x_state, dtype=torch.float32, device=device).unsqueeze(0)# [1,D]
            logits, _ = policy(ts_t, st_t)
            probs = torch.softmax(logits, dim=-1)[0]  # [sell, hold, buy]
            p_sell, p_hold, p_buy = float(probs[0]), float(probs[1]), float(probs[2])

            # Default action = HOLD
            env_action = 0

            # --- Trailing-stop exits if in position ---
            if env.position != 0 and env.entry_price is not None:
                entry = float(env.entry_price)

                if env.position == 1:
                    # LONG: track peak, arm trailing once move ≥ activate, then ratchet
                    trail_anchor = cur_px if trail_anchor is None else max(trail_anchor, cur_px)
                    move_bps = (trail_anchor / entry - 1.0) * 1e4
                    if not trail_armed and move_bps >= TRAIL_ACTIVATE_BPS:
                        trail_armed = True
                    if trail_armed:
                        desired_stop = trail_anchor * (1.0 - TRAIL_BPS / 1e-4 / 1e4)  # WRONG: fix below
                        # Correction: TRAIL_BPS is in bps, so factor is (TRAIL_BPS / 1e4)
                        desired_stop = trail_anchor * (1.0 - TRAIL_BPS / 1e4)
                        trail_stop = desired_stop if trail_stop is None else max(trail_stop, desired_stop)
                    if trail_armed and trail_stop is not None and cur_px <= trail_stop:
                        env_action = -1  # close long

                else:
                    # SHORT: track trough
                    trail_anchor = cur_px if trail_anchor is None else min(trail_anchor, cur_px)
                    move_bps = (entry / trail_anchor - 1.0) * 1e4
                    if not trail_armed and move_bps >= TRAIL_ACTIVATE_BPS:
                        trail_armed = True
                    if trail_armed:
                        desired_stop = trail_anchor * (1.0 + TRAIL_BPS / 1e4)
                        trail_stop = desired_stop if trail_stop is None else min(trail_stop, desired_stop)
                    if trail_armed and trail_stop is not None and cur_px >= trail_stop:
                        env_action = 1  # close short

            # --- Entry gating (flat & not cooling down, and no exit chosen) ---
            if env.position == 0 and env_action == 0:
                # if p_buy >= BUY_OPEN_THRESHOLD:
                if float(max(probs)) == p_buy:
                    env_action = 1   # open long
                    trail_armed = False
                    trail_anchor = cur_px
                    trail_stop = None
                # elif p_sell >= SELL_OPEN_THRESHOLD:
                elif float(max(probs)) == p_sell:
                    env_action = -1  # open short
                    trail_armed = False
                    trail_anchor = cur_px
                    trail_stop = None
                # else: remain flat

            # ------------------- DECISION SECTION (end) -------------------

            pos_before = int(env.position)
            _, reward, done, info = env.step_with_shorts(env_action)
            cumulative_pnl += reward
            equity_curve.append(cumulative_pnl)

            event = info.get("event", "hold")
            price_now = float(info.get("price", cur_px))
            realized = float(info.get("realized_pnl", 0.0))
            unreal = float(info.get("unrealized_pnl", 0.0))

            print(
                f"[VAL] Pnl={cumulative_pnl} step={env.current_step:5d} price={price_now:,.2f} "
                f"a={env_action:+d} pos={env.position:+d} event={event:<12} "
                f"real={realized:,.2f} unreal={unreal:,.2f} "
                f"PROBS sell={p_sell:.3f} hold={p_hold:.3f} buy={p_buy:.3f}",
                flush=True,
            )
            # time.sleep(2)

            # Count/log only closed trades
            if abs(realized) > 0.0:
                trades_count += 1
                ts_str = str(env.data.iloc[min(env.current_step, len(env.data)-1)].get("timestamp", ""))
                trade_log.append(ValTrade(
                    run_id=str(chunk.iloc[0].get('data_run_id')) if 'data_run_id' in chunk.columns else None,
                    episode=episode_number,
                    step_index=global_step_index,
                    timestamp=ts_str,
                    action_idx=int(torch.argmax(logits, dim=-1).item()),
                    env_action=env_action,
                    position_before=pos_before,
                    position_after=int(env.position),
                    price=price_now,
                    realized_pnl=realized,
                    unrealized_pnl=unreal,
                ))

            # Post-step hygiene
            if event in ("close_long", "close_short"):
                # start cooldown & reset trailing
                trail_armed = False
                trail_anchor = None
                trail_stop = None
            elif event in ("open_long", "open_short"):
                # reset trailing from entry price
                trail_armed = False
                trail_anchor = price_now
                trail_stop = None

            # next obs
            if not done:
                X_ts, x_state = env.get_observation_pair()

            global_step_index += 1

        # End-of-chunk auto-flatten for clean accounting
        if env.position != 0 and env.current_step >= len(env.data):
            close_action = (1 if env.position == -1 else -1)  # buy to cover short; sell to close long
            _, reward, _, info = env.step_with_shorts(close_action)
            cumulative_pnl += reward
            equity_curve.append(cumulative_pnl)
            print(
                f"[VAL] end-close price={info['price']:,.2f} "
                f"a={close_action:+d} pos=0 event={'close_short' if close_action==1 else 'close_long':<12} "
                f"real={info['realized_pnl']:,.2f} unreal={info['unrealized_pnl']:,.2f}",
                flush=True,
            )
            if abs(float(info.get("realized_pnl", 0.0))) > 0.0:
                trades_count += 1
                trade_log.append(ValTrade(
                    run_id=str(chunk.iloc[0].get('data_run_id')) if 'data_run_id' in chunk.columns else None,
                    episode=episode_number,
                    step_index=global_step_index,
                    timestamp=str(env.data.iloc[-1].get("timestamp", "")),
                    action_idx=(2 if close_action == 1 else 0),
                    env_action=close_action,
                    position_before=0,
                    position_after=0,
                    price=float(info.get("price", 0.0)),
                    realized_pnl=float(info.get("realized_pnl", 0.0)),
                    unrealized_pnl=float(info.get("unrealized_pnl", 0.0)),
                ))
                global_step_index += 1

    # ---- Metrics ----
    pnls = np.array(equity_curve, dtype=float)
    if pnls.size == 0:
        metrics = EvalMetrics(pnl=0.0, sharpe=0.0, max_drawdown=0.0, trades=0)
    else:
        running_max = np.maximum.accumulate(pnls)
        drawdowns = running_max - pnls
        max_dd = float(drawdowns.max()) if drawdowns.size else 0.0
        step_rets = np.diff(pnls) if pnls.size > 1 else np.array([0.0])
        sharpe = float((step_rets.mean() / (step_rets.std() + 1e-9)) * np.sqrt(252)) if step_rets.size else 0.0
        metrics = EvalMetrics(pnl=float(pnls[-1]), sharpe=sharpe, max_drawdown=max_dd, trades=int(trades_count))

    # ---- Save CSV with closed trades ----
    csv_path = ""
    if trade_log:
        df_trades = pd.DataFrame([t._asdict() for t in trade_log])
        csv_path = os.path.join(csv_dir, f"{csv_prefix}_ep{episode_number}.csv")
        df_trades.to_csv(csv_path, index=False)
        print(f"[VAL] per-trade CSV → {csv_path}")

    return metrics, csv_path