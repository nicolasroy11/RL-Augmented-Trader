from typing import Dict, List
from uuid import UUID
import pandas as pd
from RL.playground.stochastic.actor_critic import ActorCritic
from RL.playground.stochastic.policy_gradient import FeedForwardNN
import runtime_settings
from services.core.ML.configurations.fixture_config import DEFAULT_FEATURE_SET_ID, CONFIG_UUIDS
from services.core.models import FeatureSet, Hyperparameter, MLModel, RunConfiguration, TickData, TrainingSession

import random
import torch.nn as nn
import torch.optim as optim
from django.db.models.query import QuerySet

from services.core.dtos.policy_gradient_results_dto import EpisodeResultsDto, PolicyGradientResultsDto
import numpy as np
import torch
from services.core.ML.configurations.PPO_flattened_history.environment import Environment

DATA_TABLE_NAME = TickData._meta.db_table


class RLRepository:

    def run_ppo(self, window_size=runtime_settings.DATA_TICKS_WINDOW_LENGTH, num_episodes=100, gamma=0.99, lr=1e-4, clip_epsilon=0.2, ppo_epochs=4, batch_size=64, feature_set_id: UUID = None):
        session = TrainingSession()
        session.num_episodes = num_episodes
        run_config = RunConfiguration.objects.get(id=CONFIG_UUIDS[Environment])
        session.run_configuration = run_config
        if not feature_set_id: feature_set_id = DEFAULT_FEATURE_SET_ID
        session.feature_set = FeatureSet.objects.get(id=feature_set_id)
        
        queryset = TickData.objects.order_by('timestamp').all()
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

        hp = Hyperparameter()
        hp.ml_model = model
        hp.key = 'lr'
        hp.value = lr
        hp.save()

        hp = Hyperparameter()
        hp.ml_model = model
        hp.key = 'gamma'
        hp.value = gamma
        hp.save()

        hp = Hyperparameter()
        hp.ml_model = model
        hp.key = 'clip_epsilon'
        hp.value = clip_epsilon
        hp.save()

        hp = Hyperparameter()
        hp.ml_model = model
        hp.key = 'ppo_epochs'
        hp.value = ppo_epochs
        hp.save()

        hp = Hyperparameter()
        hp.ml_model = model
        hp.key = 'batch_size'
        hp.value = batch_size
        hp.save()
        
        input_dim = session.feature_set.get_feature_vector_size()
        self.policy = ActorCritic(input_dim, action_dim=3)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self._run_ppo(num_episodes=100, gamma=0.99, clip_epsilon=0.2, ppo_epochs=4, batch_size=64)
        

    def _run_ppo(self, num_episodes=100, gamma=0.99, clip_epsilon=0.2, ppo_epochs=4, batch_size=64) -> PolicyGradientResultsDto:
        all_episode_metrics: List[EpisodeResultsDto] = []

        for episode_number in range(num_episodes):
            print(f"\n=== Starting Episode {episode_number} ===")
            data_chunk_list = list(self.data_chunks.values())
            random.shuffle(data_chunk_list) # data chunks should be queued in different order each episode

            episode_states, episode_actions, episode_log_probs, episode_rewards, episode_values = [], [], [], [], []
            cumulative_pnl = 0.0
            episode_pnls = []

            for chunk in data_chunk_list:
                env = Environment(tick_df=chunk[:300], feature_set=self.session.feature_set)
                state = env.normalized_reset()
                done = False

                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
                    logits, value = self.policy(state_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                    next_state, reward, done, info = env.step(action.item() - 1)

                    # store as numpy to save memory
                    episode_states.append(state_tensor.detach())
                    episode_actions.append(action.detach())
                    episode_log_probs.append(log_prob.detach())
                    episode_values.append(value.detach())
                    episode_rewards.append(torch.tensor([reward], dtype=torch.float32))

                    cumulative_pnl += reward
                    episode_pnls.append(cumulative_pnl)
                    state = next_state

                    print(
                        "action probs:", dist.probs.detach().numpy(),
                        "\tcumulative_pnl:", cumulative_pnl,
                        f'\tepisode # {episode_number}'
                    )

            # Convert lists to tensors
            rewards = torch.cat(episode_rewards)
            values = torch.cat(episode_values).squeeze()
            log_probs_old = torch.cat(episode_log_probs)

            # Compute discounted returns and advantages
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32)
            advantages = returns - values

            # PPO updates
            states_tensor = torch.cat(episode_states)
            actions_tensor = torch.cat(episode_actions)

            for _ in range(ppo_epochs):
                for i in range(0, len(states_tensor), batch_size):
                    batch_states = states_tensor[i:i+batch_size]
                    batch_actions = actions_tensor[i:i+batch_size]
                    batch_returns = returns[i:i+batch_size]
                    batch_advantages = advantages[i:i+batch_size]
                    batch_log_probs_old = log_probs_old[i:i+batch_size]

                    logits, value_pred = self.policy(batch_states)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_probs_new = dist.log_prob(batch_actions)

                    ratio = (log_probs_new - batch_log_probs_old).exp()
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.MSELoss()(value_pred.squeeze(), batch_returns)
                    loss = actor_loss + 0.5 * critic_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # --- EPISODE-LEVEL METRICS ---
            episode_pnls_array = np.array(episode_pnls)
            running_max = np.maximum.accumulate(episode_pnls_array)
            drawdowns = running_max - episode_pnls_array
            max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0
            final_pnl = episode_pnls_array[-1] if len(episode_pnls_array) > 0 else 0.0
            buy_and_hold_pnl = sum(chunk.iloc[-1]['price'] - chunk.iloc[0]['price'] for chunk in data_chunk_list)
            step_returns = np.diff(episode_pnls_array)
            sharpe_ratio = (step_returns.mean() / (step_returns.std() + 1e-9)) * np.sqrt(252) if len(step_returns) > 0 else 0.0

            episode_metrics = EpisodeResultsDto()
            episode_metrics.episode_number = episode_number
            episode_metrics.final_pnl = final_pnl
            episode_metrics.episode_pnls = episode_pnls_array.tolist()
            episode_metrics.running_max = running_max.tolist()
            episode_metrics.drawdowns = drawdowns.tolist()
            episode_metrics.max_drawdown = max_drawdown
            episode_metrics.buy_and_hold_pnl = buy_and_hold_pnl
            episode_metrics.sharpe_ratio = sharpe_ratio

            all_episode_metrics.append(episode_metrics)

            model_path = f"pth_files/{self.model.id}_{episode_number}.pth"
            torch.save(self.policy.state_dict(), model_path)
            print(f"Trained policy saved to {model_path}")
            print(f"Episode {episode_number} finished. Final PnL: {final_pnl:.2f}, Buy-and-hold: {buy_and_hold_pnl:.2f}, Max Drawdown: {max_drawdown:.2f}, Sharpe: {sharpe_ratio:.2f}")

        results = PolicyGradientResultsDto()
        results.episode_results = all_episode_metrics
        return results
    

    def get_valid_data_chunks(self, queryset: QuerySet[TickData], window_size: int, min_windows: int = 10) -> Dict[UUID, List[pd.DataFrame]]:
        """
        Convert a Django ORM queryset of ticks into a list of contiguous DataFrames (regimes),
        splitting whenever a gap > 1 minute occurs between ticks.

        Parameters:
            queryset: Django ORM queryset of TickData ordered by timestamp.
            window_size: Number of steps the environment will look back.
            min_windows: Minimum number of windows per chunk to be considered valid.

        Returns:
            List of pandas DataFrames, each containing a contiguous regime of ticks.
        """
        df: pd.DataFrame = pd.DataFrame.from_records(queryset.values()).sort_values('timestamp').reset_index(drop=True)

        # Compute gaps between consecutive timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['gap'] = df['timestamp'].diff().fillna(pd.Timedelta(seconds=0))
        
        # Find split indices where gap > 1 minute
        split_indices: List[int] = df.index[df['gap'] > pd.Timedelta(minutes=1)].tolist()
        split_indices = [0] + split_indices + [len(df)]

        chunks: Dict[UUID, List[pd.DataFrame]] = {}
        for start, end in zip(split_indices[:-1], split_indices[1:]):
            chunk_df: pd.DataFrame = df.iloc[start:end].copy()
            chunk_df.drop(columns=['gap'], inplace=True)
            if len(chunk_df) >= window_size * min_windows:
                run_id = chunk_df.iloc[0]['data_run_id']
                chunks[run_id] = chunk_df.reset_index(drop=True)

        return chunks



    def run_policy_gradient(self, window_size=runtime_settings.DATA_TICKS_WINDOW_LENGTH, num_episodes=100, gamma=0.95, lr=1e-3) -> PolicyGradientResultsDto:
        queryset = TickData.objects.order_by('timestamp').all()
        data_chunks = self.get_valid_data_chunks(queryset, window_size=window_size, min_windows=10)
        if not data_chunks:
            raise ValueError("No valid chunks found with enough rows.")
        
        feature_set = FeatureSet.objects.get(id=DEFAULT_FEATURE_SET_ID)
        input_dim = feature_set.get_feature_vector_size()

        # dummy_env = StochasticSingleBuy(data_chunks[0], window_size=window_size)
        # input_dim = dummy_env.normalized_reset().shape[0]

        policy = FeedForwardNN(input_dim)
        optimizer = optim.Adam(policy.parameters(), lr=lr)

        all_episode_metrics: List[EpisodeResultsDto] = []

        for episode_number in range(num_episodes):
            print(f"\n=== Starting Episode {episode_number} ===")
            random.shuffle(data_chunks)

            log_probs = []
            rewards = []
            cumulative_pnl = 0.0
            episode_pnls = []

            for chunk in data_chunks:
                env = Environment(feature_set=feature_set, tick_df=chunk)
                current_env_state = env.normalized_reset()
                done = False

                chunk_log_probs = []
                chunk_rewards = []

                # STEP LOOP
                while not done:
                    state = torch.tensor(current_env_state, dtype=torch.float32).flatten().unsqueeze(0)
                    action_probabilities = policy(state)
                    dist = torch.distributions.Categorical(action_probabilities)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                    current_env_state, reward, done, info = env.step(action.item() - 1)

                    chunk_log_probs.append(log_prob)
                    chunk_rewards.append(reward)

                    cumulative_pnl += reward
                    episode_pnls.append(cumulative_pnl)

                    print(
                        "action probs:", action_probabilities.detach().numpy(),
                        "\tcumulative_pnl:", cumulative_pnl,
                        f'\tepisode # {episode_number}'
                    )


                # Normalize chunk rewards
                chunk_rewards = torch.tensor(chunk_rewards, dtype=torch.float32)
                if chunk_rewards.std() > 0:
                    chunk_rewards = (chunk_rewards - chunk_rewards.mean()) / (chunk_rewards.std() + 1e-9)

                rewards.extend(chunk_rewards.tolist())
                log_probs.extend(chunk_log_probs)

            # Compute discounted returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32)
            if returns.std() > 0:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)

            # Policy gradient update
            loss = 0
            for log_prob, R in zip(log_probs, returns):
                loss -= log_prob * R
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- EPISODE-LEVEL METRICS ---
            episode_pnls_array = np.array(episode_pnls)

            running_max = np.maximum.accumulate(episode_pnls_array)
            drawdowns = running_max - episode_pnls_array
            max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0

            final_pnl = episode_pnls_array[-1] if len(episode_pnls_array) > 0 else 0.0

            # Buy-and-hold cumulative PnL
            buy_and_hold_pnl = sum(chunk.iloc[-1]['price'] - chunk.iloc[0]['price'] for chunk in data_chunks)

            # Sharpe ratio (assuming step-level returns)
            step_returns = np.diff(episode_pnls_array)
            sharpe_ratio = (step_returns.mean() / (step_returns.std() + 1e-9)) * np.sqrt(252) if len(step_returns) > 0 else 0.0

            episode_metrics = EpisodeResultsDto()
            episode_metrics.episode_number = episode_number
            episode_metrics.final_pnl = final_pnl
            episode_metrics.episode_pnls = episode_pnls_array.tolist()
            episode_metrics.running_max = running_max.tolist()
            episode_metrics.drawdowns = drawdowns.tolist()
            episode_metrics.max_drawdown = max_drawdown
            episode_metrics.buy_and_hold_pnl = buy_and_hold_pnl
            episode_metrics.sharpe_ratio = sharpe_ratio

            all_episode_metrics.append(episode_metrics)

            model_path = f"pth_files/trained_policy_gradient_{episode_number}_.pth"
            torch.save(policy.state_dict(), model_path)
            print(f"Trained policy saved to {model_path}")

            print(f"Episode {episode_number} finished. Final PnL: {final_pnl:.2f}, Buy-and-hold: {buy_and_hold_pnl:.2f}, Max Drawdown: {max_drawdown:.2f}, Sharpe: {sharpe_ratio:.2f}")

        results = PolicyGradientResultsDto()
        results.episode_results = all_episode_metrics

        return results
    
