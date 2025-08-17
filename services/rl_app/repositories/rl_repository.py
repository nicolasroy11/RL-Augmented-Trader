from typing import List

import pandas as pd
from RL.playground.stochastic.policy_gradient import FeedForwardNN
from services.core.models import BTCFDUSDData, BTCFDUSDTick
from binance.client import Client

from typing import List
import matplotlib.pyplot as plt
import random
import numpy as np
from RL.playground.stochastic.policy_gradient import FeedForwardNN
import torch
import torch.optim as optim
from django.db.models.query import QuerySet

from services.rl_app.environments.stochastic_single_buy_sell import StochasticSingleBuy

TICKS_TABLE_NAME = BTCFDUSDTick._meta.db_table
DATA_TABLE_NAME = BTCFDUSDData._meta.db_table
client = Client(tld='com')
store_frequency_secs = 5


class RLRepository():

    def run_policy_gradient(self, window_size=150, num_episodes=100, gamma=0.95, lr=1e-3):
        """
        Train a policy gradient agent across multiple discontinuous market datasets (pgsql data chunks).
        PnL is continuous across all files in a single episode.
        """

        # STEP 1 — Gather all valid DuckDB files
        queryset = BTCFDUSDData.objects.order_by('timestamp').all()
        data_chunks = self.get_valid_data_chunks(queryset, window_size=window_size, min_windows=10)
        if not data_chunks:
            raise ValueError("No valid chunks files found with enough rows.")

        # STEP 2 — Initialize dummy env to determine observation dimension (num of features)
        dummy_env = StochasticSingleBuy(data=data_chunks[0], window_size=window_size)
        dummy_obs = dummy_env.normalized_reset()
        input_dim = dummy_obs.shape[0]  # Flattened observation vector

        # STEP 3 — Initialize policy network and optimizer
        policy = FeedForwardNN(input_dim)
        optimizer = optim.Adam(policy.parameters(), lr=lr)

        all_episode_pnls = []  # Stores cumulative PnL curves for plotting

        # STEP 4 — Main training loop
        for episode_number in range(num_episodes):
            print(f"\n=== Starting Episode {episode_number + 1} ===")
            random.shuffle(data_chunks)  # Shuffle file order to avoid memorization

            log_probs = []
            rewards = []
            cumulative_pnl = 0.0   # Continuous PnL across all files in this episode
            episode_pnls = []

            # STEP 5 — Loop through each file
            for chunk in data_chunks:

                # Initialize environment for this file
                env = StochasticSingleBuy(chunk, window_size=window_size)

                # Reset env but preserve current cumulative PnL
                current_env_state = env.normalized_reset()
                done = False

                file_log_probs = []
                file_rewards = []

                # STEP 5a — Interact with environment until this file is done
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

                    # Track file-specific and episode-wide metrics
                    file_log_probs.append(log_prob)
                    file_rewards.append(reward)

                    cumulative_pnl += reward
                    episode_pnls.append(cumulative_pnl)

                    print(
                        "action probs:", action_probabilities.detach().numpy(),
                        "\tcumulative_pnl:", cumulative_pnl,
                        f'\tepisode # {episode_number}'
                    )

                # STEP 5b — Normalize rewards for stability
                file_rewards = torch.tensor(file_rewards, dtype=torch.float32)
                if file_rewards.std() > 0:
                    file_rewards = (file_rewards - file_rewards.mean()) / (file_rewards.std() + 1e-9)

                # Append normalized rewards and log_probs to episode-level lists
                rewards.extend(file_rewards.tolist())
                log_probs.extend(file_log_probs)

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
            print(f"Episode {episode_number} finished. Total PnL: {cumulative_pnl:.2f}")

        # STEP 8 — Plot cumulative PnL curves
        plt.figure(figsize=(12, 6))
        # for i, ep_pnl in enumerate(all_episode_pnls):
        #     plt.plot(range(len(ep_pnl)), ep_pnl, label=f"Episode {i+1}")
        for i, episode_pnl in enumerate(all_episode_pnls):
            if i in [0, int(np.floor(len(all_episode_pnls)/2)), len(all_episode_pnls) - 1]:
                plt.plot(range(len(episode_pnl)), episode_pnl, label=f"Episode {i}")
        plt.xlabel("Step within Episode")
        plt.ylabel("Cumulative PnL")
        plt.title("PnL Progression Across Episodes (Multi-file Policy Gradient)")
        plt.legend()
        plt.show()
        t = 0


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
