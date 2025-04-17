import hashlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from vis_gym import *

# =============================================================================
# Imports from your CNN-DQN implementation file
# Make sure this file is in the same directory or adjust the import path accordingly.
from CNN import Config, setup_environment, CNNDQNAgent


# =============================================================================

# --- EVALUATION & PLOTTING FOR CNN-DQN AGENT ---

def evaluate_and_plot(model_path: str,
                      num_episodes: int = None,
                      ma_window: int = 10):
    """
    Load a trained CNN-DQN agent, run evaluation episodes, and produce high-quality plots:
      1) Episode reward + moving average + mean
      2) Histogram of rewards
      3) Boxplot of rewards
    """
    # Initialize environment and agent
    env, n_actions = setup_environment()
    agent = CNNDQNAgent(env, n_actions)
    agent.load_model(model_path)
    agent.policy_net.eval()

    # Determine how many episodes to run
    if num_episodes is None:
        num_episodes = Config.EVAL_EPISODES

    rewards = []
    for ep in range(1, num_episodes + 1):
        obs, _, done, _ = env.reset()
        state = agent.get_state_tensor(reset=True)
        ep_reward = 0

        while not done:
            # Greedy action selection
            with torch.no_grad():
                action = agent.policy_net(state).max(1).indices.view(1, 1)

            obs, r, done, _ = env.step(action)
            ep_reward += r
            state = agent.get_state_tensor()

        rewards.append(ep_reward)
        if ep == 1 or ep % max(1, num_episodes // 10) == 0:
            print(f"Episode {ep:3d}/{num_episodes}: Reward = {ep_reward:.2f}")

    env.close()

    # Convert to NumPy
    rewards = np.array(rewards)
    mean_r = rewards.mean()
    std_r = rewards.std()
    median_r = np.median(rewards)
    print(f"\nSummary over {num_episodes} episodes: ")
    print(f"  Mean   = {mean_r:.2f}")
    print(f"  Std    = {std_r:.2f}")
    print(f"  Median = {median_r:.2f}\n")

    # Compute moving average
    if len(rewards) >= ma_window:
        kernel = np.ones(ma_window) / ma_window
        ma = np.convolve(rewards, kernel, mode='valid')
        ma = np.concatenate([np.full(ma_window - 1, np.nan), ma])
    else:
        ma = np.full_like(rewards, np.nan)

    # Styling parameters
    dpi = 300
    label_fs = 14
    title_fs = 16
    tick_fs = 12
    line_lw = 2
    grid_kw = dict(linestyle='--', alpha=0.5)

    # 1) Reward curve + moving average + mean line
    fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
    ax.plot(rewards, alpha=0.3, linewidth=1, label='Episode reward')
    ax.plot(ma, linewidth=line_lw, label=f'{ma_window}-episode MA')
    ax.axhline(mean_r, linestyle='--', linewidth=1,
               label=f'Overall mean = {mean_r:.2f}')
    ax.set_xlabel('Episode', fontsize=label_fs)
    ax.set_ylabel('Total Reward', fontsize=label_fs)
    ax.set_title('CNN-DQN Evaluation: Reward per Episode', fontsize=title_fs)
    ax.legend(fontsize=label_fs)
    ax.grid(**grid_kw)
    ax.tick_params(labelsize=tick_fs)
    plt.tight_layout()
    fig.savefig('cnn_reward_curve.png')
    plt.show()

    # 2) Histogram of rewards
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    ax.hist(rewards, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Episode Total Reward', fontsize=label_fs)
    ax.set_ylabel('Frequency', fontsize=label_fs)
    ax.set_title('CNN-DQN Reward Distribution', fontsize=title_fs)
    ax.grid(**grid_kw)
    ax.tick_params(labelsize=tick_fs)
    plt.tight_layout()
    fig.savefig('cnn_reward_histogram.png')
    plt.show()

    # 3) Boxplot of rewards
    fig, ax = plt.subplots(figsize=(4, 6), dpi=dpi)
    ax.boxplot(rewards, vert=True, patch_artist=True,
               boxprops=dict(facecolor='white', edgecolor='black'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_ylabel('Episode Total Reward', fontsize=label_fs)
    ax.set_title('CNN-DQN Reward Boxplot', fontsize=title_fs)
    ax.grid(axis='y', **grid_kw)
    ax.tick_params(labelsize=tick_fs)
    plt.tight_layout()
    fig.savefig('cnn_reward_boxplot.png')
    plt.show()


if __name__ == '__main__':
    # Replace with your actual model filename
    model_file = 'cnn_dqn_model_20250404-182737.pth'
    evaluate_and_plot(model_path=model_file)
