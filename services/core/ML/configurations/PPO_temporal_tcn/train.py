from typing import Dict, List, Tuple
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

from services.core.dtos.policy_gradient_results_dto import EpisodeResultsDto, PolicyGradientResultsDto
import numpy as np

# Two-input env & policy
from services.core.ML.configurations.PPO_temporal_tcn.environment import (
    Environment as EnvironmentTCN,
    TCNActorCriticWithState,
)

DATA_TABLE_NAME = TickData._meta.db_table


def compute_gae(
    rewards: torch.Tensor,      # [N]
    values: torch.Tensor,       # [N]
    dones: torch.Tensor,        # [N] booleans {0,1}
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation.
    Returns:
        advantages: [N]
        returns:    [N] = advantages + values
    """
    N = rewards.shape[0]
    advantages = torch.zeros(N, dtype=torch.float32, device=rewards.device)
    last_adv = 0.0
    for t in reversed(range(N)):
        next_nonterminal = 1.0 - (dones[t].item() if t < N else 1.0)
        next_value = values[t + 1] if t + 1 < N else torch.tensor(0.0, device=values.device, dtype=torch.float32)
        delta = rewards[t] + gamma * next_value * (1.0 - (dones[t + 1] if t + 1 < N else 0.0)) - values[t]
        last_adv = delta + gamma * lam * (1.0 - (dones[t + 1] if t + 1 < N else 0.0)) * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


class RLRepository:

    def run_ppo(
        self,
        window_size=runtime_settings.DATA_TICKS_WINDOW_LENGTH,
        num_episodes=100,
        gamma=0.99,
        gae_lambda=0.95,
        lr=3e-4,
        clip_epsilon=0.2,
        ppo_epochs=4,
        batch_size=64,
        ent_coef=0.05,
        feature_set_id: UUID = None,
        is_futures = False
    ):
        session = TrainingSession()
        session.num_episodes = num_episodes

        # point the run config at the TCN environment class
        run_config = RunConfiguration.objects.get(id=CONFIG_UUIDS[EnvironmentTCN])
        session.run_configuration = run_config

        if not feature_set_id:
            feature_set_id = DEFAULT_FEATURE_SET_ID
        session.feature_set = FeatureSet.objects.get(id=feature_set_id)

        queryset = TickData.objects.order_by('timestamp').filter(data_run__is_futures=is_futures).all()
        self.data_chunks = self.get_valid_data_chunks(queryset, window_size=window_size, min_windows=10)
        if not self.data_chunks:
            raise ValueError("No valid chunks found with enough rows.")

        session.save()
        session.data_runs.set(list(self.data_chunks.keys()))
        self.session = session

        model = MLModel()
        model.feature_set = session.feature_set
        model.run_configuration = session.run_configuration
        model.save()
        self.model = model

        # log hyperparams
        for k, v in {
            "lr": lr,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_epsilon": clip_epsilon,
            "ppo_epochs": ppo_epochs,
            "batch_size": batch_size,
            "ent_coef": ent_coef,
        }.items():
            Hyperparameter(ml_model=model, key=k, value=v).save()

        # Lazy-build two-input TCN policy after we see first (T,F_ts) and D
        self.policy: TCNActorCriticWithState | None = None
        self.optimizer = None

        self._run_ppo(num_episodes, gamma, gae_lambda, clip_epsilon, ppo_epochs, batch_size, ent_coef, lr)


    def _run_ppo(self, num_episodes, gamma, gae_lambda, clip_epsilon, ppo_epochs, batch_size, ent_coef, lr) -> PolicyGradientResultsDto:
        all_episode_metrics: List[EpisodeResultsDto] = []

        device = torch.device("cpu")


        for episode_number in range(num_episodes):
            print(f"\n=== Starting Episode {episode_number} ===")
            data_chunk_list = list(self.data_chunks.values())
            random.shuffle(data_chunk_list)  # different order each episode

            # buffers
            ts_buf: List[torch.Tensor] = []        # [T, F_ts]
            state_buf: List[torch.Tensor] = []     # [D]
            action_buf: List[torch.Tensor] = []    # scalar indices (0/1/2)
            logp_buf: List[torch.Tensor] = []
            value_buf: List[torch.Tensor] = []
            reward_buf: List[torch.Tensor] = []
            done_buf: List[torch.Tensor] = []
            action_probs_log: List[List[float]] = []

            cumulative_pnl = 0.0
            running_episode_pnls: List[float] = []
            prices = []

            for chunk in data_chunk_list:
                step = 0
                prices.extend(list(chunk['price']))
                env = EnvironmentTCN(tick_df=chunk, feature_set=self.session.feature_set)
                env.normalized_reset()

                X_ts, x_state = env.get_observation_pair()  # first pair

                # build policy lazily on first pair
                if self.policy is None:
                    T, F_ts = X_ts.shape
                    D = x_state.shape[-1]
                    self.policy = TCNActorCriticWithState(in_feats_ts=F_ts, state_dim=D, action_dim=3).to(device)
                    self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

                done = False
                while not done:
                    step += 1
                    ts_t = torch.tensor(X_ts, dtype=torch.float32, device=device).unsqueeze(0)       # [1,T,F_ts]
                    st_t = torch.tensor(x_state, dtype=torch.float32, device=device).unsqueeze(0)    # [1,D]

                    logits, value = self.policy(ts_t, st_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()                      # 0,1,2
                    log_prob = dist.log_prob(action)

                    # map to {-1,0,1} for env
                    a_idx = int(action.item())
                    env_action = (-1 if a_idx == 0 else (0 if a_idx == 1 else 1))

                    _, reward, done, info = env.step_with_shorts(env_action)

                    # store step
                    ts_buf.append(ts_t.squeeze(0).detach())        # [T,F_ts]
                    state_buf.append(st_t.squeeze(0).detach())     # [D]
                    action_buf.append(action.detach())             # scalar idx
                    logp_buf.append(log_prob.detach())
                    value_buf.append(value.detach().squeeze(-1))   # [1] -> store as [1]
                    reward_buf.append(torch.tensor([reward], dtype=torch.float32, device=device))
                    done_buf.append(torch.tensor([1.0 if done else 0.0], dtype=torch.float32, device=device))
                    action_probs_log.append(list(dist.probs.detach().cpu().numpy().flatten()))

                    cumulative_pnl += reward
                    running_episode_pnls.append(cumulative_pnl)

                    # next observation pair
                    if not done:
                        X_ts, x_state = env.get_observation_pair()

                    print(
                        "step:", step,
                        "\taction probs:", dist.probs.detach().cpu().numpy(),
                        "\tcumulative_pnl:", cumulative_pnl,
                        f'\tepisode # {episode_number}'
                    )

            # --- tensors ---
            rewards = torch.cat(reward_buf).squeeze(-1)            # [N]
            values = torch.cat(value_buf).squeeze(-1)              # [N]
            log_probs_old = torch.cat(logp_buf)                    # [N]
            ts_tensor = torch.stack(ts_buf, dim=0)                 # [N, T, F_ts]
            state_tensor = torch.stack(state_buf, dim=0)           # [N, D]
            actions_tensor = torch.cat(action_buf)                 # [N]
            dones = torch.cat(done_buf).squeeze(-1)                # [N], 1.0 where done, else 0.0

            # --- GAE advantages & returns ---
            advantages, returns = compute_gae(rewards, values, dones, gamma=gamma, lam=gae_lambda)
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # --- PPO updates ---
            N = ts_tensor.shape[0]
            for _ in range(ppo_epochs):
                # one simple sequential pass; you can shuffle indices if desired
                for i in range(0, N, batch_size):
                    j = slice(i, i + batch_size)
                    batch_ts = ts_tensor[j]                      # [B,T,F_ts]
                    batch_state = state_tensor[j]                # [B,D]
                    batch_actions = actions_tensor[j]            # [B]
                    batch_returns = returns[j]                   # [B]
                    batch_advantages = advantages[j]             # [B]
                    batch_log_probs_old = log_probs_old[j]       # [B]

                    logits, value_pred = self.policy(batch_ts, batch_state)  # value_pred: [B,1]
                    dist = torch.distributions.Categorical(logits=logits)
                    log_probs_new = dist.log_prob(batch_actions)             # [B]

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

            # --- EPISODE METRICS ---
            episode_pnls_array = np.array(running_episode_pnls)
            running_max = np.maximum.accumulate(episode_pnls_array) if episode_pnls_array.size else np.array([])
            drawdowns = (running_max - episode_pnls_array) if episode_pnls_array.size else np.array([])
            max_drawdown = float(drawdowns.max()) if drawdowns.size else 0.0
            final_pnl = float(episode_pnls_array[-1]) if episode_pnls_array.size else 0.0
            buy_and_hold_pnl = sum(chunk.iloc[-1]['price'] - chunk.iloc[0]['price'] for chunk in data_chunk_list)
            step_returns = np.diff(episode_pnls_array) if episode_pnls_array.size else np.array([])
            sharpe_ratio = (step_returns.mean() / (step_returns.std() + 1e-9) * np.sqrt(252)) if step_returns.size else 0.0

            episode_metrics = EpisodeResultsDto()
            episode_metrics.episode_number = episode_number
            episode_metrics.final_pnl = final_pnl
            episode_metrics.episode_pnls = episode_pnls_array.tolist()
            episode_metrics.running_max = running_max.tolist() if running_max.size else []
            episode_metrics.drawdowns = drawdowns.tolist() if drawdowns.size else []
            episode_metrics.max_drawdown = max_drawdown
            episode_metrics.buy_and_hold_pnl = buy_and_hold_pnl
            episode_metrics.sharpe_ratio = sharpe_ratio
            episode_metrics.action_probs = action_probs_log
            episode_metrics.prices = prices
            all_episode_metrics.append(episode_metrics)

            model_path = f"pth_files/{self.model.id}_tcn2in_gae_{episode_number}.pth"
            torch.save(self.policy.state_dict(), model_path)
            print(f"Trained two-input TCN policy (GAE) saved to {model_path}")
            print(f"Episode {episode_number} finished. Final PnL: {final_pnl:.2f}, "
                  f"Buy-and-hold: {buy_and_hold_pnl:.2f}, Max Drawdown: {max_drawdown:.2f}, Sharpe: {sharpe_ratio:.2f}")

            # will overwrite at every episode
            json_log_training_progression(all_episode_metrics, f'PPO_TCN_2in_GAE_{self.session.id}')

        results = PolicyGradientResultsDto()
        results.episode_results = all_episode_metrics
        return results


    def get_valid_data_chunks(self, queryset: QuerySet[TickData], window_size: int, min_windows: int = 10) -> Dict[UUID, pd.DataFrame]:
        """
        Separate ticks chunks into contiguous DataFrames (regimes), splitting when gap > 1 minute.
        Keeps chunks with at least `window_size * min_windows` rows.
        Returns a dict keyed by data_run_id -> chunk dataframe.
        """
        df: pd.DataFrame = pd.DataFrame.from_records(queryset.values()).sort_values('timestamp').reset_index(drop=True)

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
                chunks[run_id] = chunk_df.reset_index(drop=True)
        return chunks
