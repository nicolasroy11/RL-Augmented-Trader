from datetime import datetime, timezone
from typing import List
import pandas as pd
from RL.playground.stochastic.policy_gradient import FeedForwardNN
from services.core.models import BTCFDUSDData, BTCFDUSDTick

from typing import List
import random
import numpy as np
from RL.playground.stochastic.policy_gradient import FeedForwardNN
import torch
import torch.optim as optim
from django.db.models.query import QuerySet

from services.core.dtos.policy_gradient_results_dto import PolicyGradientResultsDto
from services.rl_app.environments.stochastic_single_buy import StochasticSingleBuy

TICKS_TABLE_NAME = BTCFDUSDTick._meta.db_table
DATA_TABLE_NAME = BTCFDUSDData._meta.db_table


class RLRepository():

    def run_policy_gradient(self, window_size=150, num_episodes=100, gamma=0.95, lr=1e-3) -> PolicyGradientResultsDto:
        """
        Train a policy gradient agent across multiple discontinuous market datasets (pgsql data chunks).
        PnL is continuous across all data chunks in a single episode.
        """

        # STEP 1 — Gather all valid data chunks
        queryset = BTCFDUSDData.objects.order_by('timestamp').all()
        data_chunks = self.get_valid_data_chunks(queryset, window_size=window_size, min_windows=10)
        if not data_chunks:
            raise ValueError("No valid chunks found with enough rows.")

        # STEP 2 — Initialize dummy env to determine observation dimension (num of features)
        dummy_env = StochasticSingleBuy(data=data_chunks[0], window_size=window_size)
        dummy_obs = dummy_env.normalized_reset()
        input_dim = dummy_obs.shape[0]  # Flattened observation vector

        # STEP 3 — Initialize policy network and optimizer
        policy = FeedForwardNN(input_dim)
        optimizer = optim.Adam(policy.parameters(), lr=lr)

        all_episode_pnls = []  # Stores cumulative PnL curves for plotting
        final_episode_pnls = []

        # STEP 4 — Main training loop
        for episode_number in range(num_episodes - 1):
            print(f"\n=== Starting Episode {episode_number} ===")
            random.shuffle(data_chunks)  # Shuffle chunk order to avoid memorization

            log_probs = []
            rewards = []
            cumulative_pnl = 0.0   # Continuous PnL across all chunks in this episode
            episode_pnls = []

            # STEP 5 — Loop through each chunk
            for chunk in data_chunks:

                # Initialize environment for this chunk
                env = StochasticSingleBuy(chunk, window_size=window_size)

                # Reset env but preserve current cumulative PnL
                current_env_state = env.normalized_reset()
                done = False

                chunk_log_probs = []
                chunk_rewards = []

                # STEP 5a — Interact with environment until this chunk is done
                while not done:
                    # Convert observation to tensor and flatten
                    state = torch.tensor(current_env_state, dtype=torch.float32).flatten().unsqueeze(0)

                    # Forward pass through policy
                    action_probabilities = policy(state)
                    dist = torch.distributions.Categorical(action_probabilities)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                    # Step environment (actions: -1,0,1)
                    current_env_state, reward, done, info = env.step(action.item() - 1)

                    # Track chunk-specific and episode-wide metrics
                    chunk_log_probs.append(log_prob)
                    chunk_rewards.append(reward)

                    cumulative_pnl += reward
                    episode_pnls.append(cumulative_pnl)

                    print(
                        "action probs:", action_probabilities.detach().numpy(),
                        "\tcumulative_pnl:", cumulative_pnl,
                        f'\tepisode # {episode_number}'
                    )

                # STEP 5b — Normalize rewards for stability
                chunk_rewards = torch.tensor(chunk_rewards, dtype=torch.float32)
                if chunk_rewards.std() > 0:
                    chunk_rewards = (chunk_rewards - chunk_rewards.mean()) / (chunk_rewards.std() + 1e-9)

                # Append normalized rewards and log_probs to episode-level lists
                rewards.extend(chunk_rewards.tolist())
                log_probs.extend(chunk_log_probs)

            # STEP 6 — Compute discounted returns for the full episode
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32)
            if returns.std() > 0:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)

            # STEP 7 — Policy gradient update
            loss = 0
            for log_prob, R in zip(log_probs, returns):
                loss -= log_prob * R
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_episode_pnls.append(episode_pnls)
            final_episode_pnls.append(episode_pnls[-1])  # store final PnL
            print(f"Episode {episode_number} finished. Total PnL: {cumulative_pnl:.2f}")

        # Save trained policy
        model_path = f"pth_files/trained_policy_{datetime.now(tz=timezone.utc)}.pth"
        torch.save(policy.state_dict(), model_path)
        print(f"Trained policy saved to {model_path}")

        results = PolicyGradientResultsDto()
        results.all_episode_final_pnls = final_episode_pnls

        return results


    def get_valid_data_chunks(self, queryset: QuerySet[BTCFDUSDData], window_size: int, min_windows: int = 10) -> List[pd.DataFrame]:
        """
        Convert a Django ORM queryset of ticks into a list of contiguous DataFrames (regimes),
        splitting whenever a gap > 1 minute occurs between ticks.

        Parameters:
            queryset: Django ORM queryset of BTCFDUSDData ordered by timestamp.
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

        chunks: List[pd.DataFrame] = []
        for start, end in zip(split_indices[:-1], split_indices[1:]):
            chunk_df: pd.DataFrame = df.iloc[start:end].copy()
            chunk_df.drop(columns=['gap'], inplace=True)
            if len(chunk_df) >= window_size + min_windows:
                chunks.append(chunk_df.reset_index(drop=True))

        return chunks
