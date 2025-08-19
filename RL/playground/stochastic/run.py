from typing import List
import matplotlib.pyplot as plt
import random
import numpy as np
from RL.playground.stochastic.policy_gradient import FeedForwardNN
from RL.playground.stochastic.env import TradingEnvWithPnL
import torch
import torch.optim as optim
from db._helpers import get_valid_duckdb_files
from db.settings import DUCKDB_ARCHIVES_PATH, DUCKDB_FILE_PATH

# from trading_env import TradingEnvWithPnL  # adjust import if needed

import os
import glob

import runtime_settings


# for reproduceable results
seed = 77
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def run_stochastic_episodes(db_path, window_size=10, num_episodes=20):
    all_episode_pnls = []

    for ep in range(num_episodes):
        env = TradingEnvWithPnL(db_path, window_size=window_size)
        obs = env.reset()
        done = False

        cumulative_pnl = 0.0
        episode_pnls = []

        print(f"Starting Episode {ep + 1}")

        while not done:
            # Random action example: -1 = sell, 0 = hold, 1 = buy
            action = random.choice([-1, 0, 1])
            obs, reward, done, info = env.step(action)

            cumulative_pnl += reward
            episode_pnls.append(cumulative_pnl)

        all_episode_pnls.append(episode_pnls)
        print(f"Episode {ep + 1} finished. Total PnL: {cumulative_pnl:.2f}")

    # Plot PnL for all episodes
    plt.figure(figsize=(12, 6))
    for i, ep_pnl in enumerate(all_episode_pnls):
        plt.plot(range(len(ep_pnl)), ep_pnl, label=f"Episode {i+1}")
    plt.xlabel("Step within Episode")
    plt.ylabel("Cumulative PnL")
    plt.title("PnL Progression Across Episodes")
    plt.legend()
    plt.show()


def run_policy_gradient(db_path, window_size=10, num_episodes=20, gamma=0.99, lr=1e-3):
    # Initialize dummy env to get observation shape
    dummy_env = TradingEnvWithPnL(db_path, window_size=window_size)
    dummy_obs = dummy_env.normalized_reset()
    input_dim = dummy_obs.shape[0]

    # Initialize policy and optimizer once
    policy = FeedForwardNN(input_dim)
    for name, param in policy.named_parameters():
        print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}, max={param.data.max():.4f}, min={param.data.min():.4f}")

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    all_episode_pnls: List[List[float]] = []

    for ep in range(num_episodes):
        # New environment per episode
        env = TradingEnvWithPnL(db_path=db_path, window_size=window_size)
        current_env_state = env.normalized_reset()
        done = False

        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        cumulative_pnl: float = 0.0
        episode_pnls: List[float] = []

        print(f"Starting Episode {ep + 1}")

        while not done:
            # Flatten observation
            state = torch.tensor(data=current_env_state, dtype=torch.float32).flatten().unsqueeze(0)  # shape: (1, input_dim)
            action_probabilities = policy(state)
            dist = torch.distributions.Categorical(probs=action_probabilities)
            action = dist.sample()
            log_prob: torch.Tensor = dist.log_prob(action)

            current_env_state, reward, done, info = env.step(action.item() - 1)  # map 0,1,2 → -1,0,1

            log_probs.append(log_prob)
            rewards.append(reward)

            cumulative_pnl += reward
            print("action probs:", action_probabilities.detach().numpy(), "\tcumulative_pnl: ", cumulative_pnl, f'\tepisode # {ep}')
            episode_pnls.append(cumulative_pnl)

        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize

        # Policy gradient update
        loss = 0
        for log_prob, R in zip(log_probs, returns):
            loss -= log_prob * R
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_episode_pnls.append(episode_pnls)
        print(f"Episode {ep + 1} finished. Total PnL: {cumulative_pnl:.2f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    # for i, ep_pnl in enumerate(all_episode_pnls):
    #     plt.plot(range(len(ep_pnl)), ep_pnl, label=f"Episode {i+1}")
    for i, episode_pnl in enumerate(all_episode_pnls):
        if i in [0, int(np.floor(len(all_episode_pnls)/2)), len(all_episode_pnls)]:
            plt.plot(range(len(episode_pnl)), episode_pnl, label=f"Episode {i+1}")
    plt.xlabel("Step within Episode")
    plt.ylabel("Cumulative PnL")
    plt.title("PnL Progression Across Episodes (Policy Gradient)")
    plt.legend()
    plt.show()
    t = 0


def run_multi_file_policy_gradient(folder_path, window_size=10, num_episodes=20, gamma=0.95, lr=1e-3):
    """
    Train a policy gradient agent across multiple discontinuous market datasets (archive duckdb files).
    PnL is continuous across all files in a single episode.
    """
    min_amount_of_windows_in_file = 10  # Minimum number of windows per file

    # STEP 1 — Gather all valid DuckDB files
    duckdb_files = get_valid_duckdb_files(folder_path, window_size, min_amount_of_windows_in_file)
    if not duckdb_files:
        raise ValueError("No valid DuckDB files found with enough rows.")

    # STEP 2 — Initialize dummy env to determine observation dimension
    dummy_env = TradingEnvWithPnL(duckdb_files[0], window_size=window_size)
    dummy_obs = dummy_env.normalized_reset()
    input_dim = dummy_obs.shape[0]  # Flattened observation vector

    # STEP 3 — Initialize policy network and optimizer
    policy = FeedForwardNN(input_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    all_episode_pnls = []  # Stores cumulative PnL curves for plotting

    # STEP 4 — Main training loop
    for episode_number in range(num_episodes):
        print(f"\n=== Starting Episode {episode_number + 1} ===")
        random.shuffle(duckdb_files)  # Shuffle file order to avoid memorization

        log_probs = []
        rewards = []
        cumulative_pnl = 0.0   # Continuous PnL across all files in this episode
        episode_pnls = []

        # STEP 5 — Loop through each file
        for file_path in duckdb_files:
            print(f"[Episode {episode_number}] Training on file: {os.path.basename(file_path)}")

            # Initialize environment for this file
            env = TradingEnvWithPnL(file_path, window_size=window_size)

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


if __name__ == "__main__":

    # TODO: Make these proportional to the size of training data
    window_size = runtime_settings.DATA_TICKS_WINDOW
    num_episodes = 100
    
    # print("\n=== Running pure random baseline ===\n")
    # run_stochastic_episodes(db_path, window_size, num_episodes)

    # print("\n=== Running policy gradient training ===\n")
    # run_policy_gradient(DUCKDB_FILE_PATH, window_size, num_episodes)

    print("\n=== Running policy gradient training ===\n")
    run_multi_file_policy_gradient(DUCKDB_ARCHIVES_PATH, window_size, num_episodes)
