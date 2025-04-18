import hashlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from vis_gym import *

from CNN import Config, setup_environment, CNNDQNAgent


def evaluate_and_plot(model_path: str,
                      num_episodes: int = None,
                      ma_window: int = 10):
    """
    Load a trained CNN-DQN agent, run evaluation episodes, and produce high-quality plots:
      1) Reward curve + moving average + mean
      2) Histogram of rewards
      3) Boxplot of rewards
      4) Steps curve + moving average + mean
    """
    # --- init ---
    env, n_actions = setup_environment()
    agent = CNNDQNAgent(env, n_actions)
    agent.load_model(model_path)
    agent.policy_net.eval()

    if num_episodes is None:
        num_episodes = Config.EVAL_EPISODES

    rewards = []
    steps = []

    # --- run episodes ---
    for ep in range(1, num_episodes + 1):
        obs, _, done, _ = env.reset()
        state = agent.get_state_tensor(reset=True)
        ep_reward = 0
        ep_steps = 0

        while not done:
            with torch.no_grad():
                action = agent.policy_net(state).max(1).indices.view(1, 1)
            obs, r, done, _ = env.step(action)
            ep_reward += r
            ep_steps += 1
            state = agent.get_state_tensor()

        rewards.append(ep_reward)
        steps.append(ep_steps)

        if ep == 1 or ep % max(1, num_episodes // 10) == 0:
            print(f"Episode {ep:3d}/{num_episodes}: "
                  f"Reward = {ep_reward:.2f}, Steps = {ep_steps}")

    env.close()

    # --- summary stats ---
    rewards = np.array(rewards)
    steps = np.array(steps)

    mean_r, std_r, median_r = rewards.mean(), rewards.std(), np.median(rewards)
    mean_s, std_s, median_s = steps.mean(), steps.std(), np.median(steps)
    print(f"\nOver {num_episodes} episodes:")
    print(f"  Reward → mean: {mean_r:.2f}, std: {std_r:.2f}, median: {median_r:.2f}")
    print(f"  Steps  → mean: {mean_s:.2f}, std: {std_s:.2f}, median: {median_s:.2f}\n")

    # --- moving averages ---
    def moving_avg(x):
        if len(x) >= ma_window:
            k = np.ones(ma_window) / ma_window
            ma = np.convolve(x, k, mode='valid')
            return np.concatenate([np.full(ma_window - 1, np.nan), ma])
        else:
            return np.full_like(x, np.nan)

    ma_r = moving_avg(rewards)
    ma_s = moving_avg(steps)

    # --- stylistic params ---
    dpi = 300
    label_fs, title_fs, tick_fs = 14, 16, 12
    line_lw = 2
    grid_kw = dict(linestyle='--', alpha=0.5)

    # 1) Reward curve
    low_r, high_r = np.percentile(rewards, [1, 99])
    fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
    ax.plot(rewards, alpha=0.3, linewidth=1, label='Episode reward')
    # ax.plot(ma_r, linewidth=line_lw, label=f'{ma_window}-ep MA')
    ax.axhline(mean_r, color='red', linestyle='--', linewidth=1,
               label=f'Overall mean = {mean_r:.2f}')
    ax.set_ylim(low_r, high_r)
    ax.set_xlabel('Episode', fontsize=label_fs)
    ax.set_ylabel('Total Reward', fontsize=label_fs)
    ax.set_title('CNN-DQN Evaluation: Reward per Episode', fontsize=title_fs)
    ax.legend(fontsize=label_fs)
    ax.grid(**grid_kw)
    ax.tick_params(labelsize=tick_fs)
    plt.tight_layout()
    fig.savefig('cnn_reward_curve.png')
    plt.show()

    # # 2) Histogram of rewards
    # fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    # ax.hist(rewards, bins=20, alpha=0.7, edgecolor='black')
    # ax.set_xlabel('Episode Total Reward', fontsize=label_fs)
    # ax.set_ylabel('Frequency', fontsize=label_fs)
    # ax.set_title('CNN-DQN Reward Distribution', fontsize=title_fs)
    # ax.grid(**grid_kw)
    # ax.tick_params(labelsize=tick_fs)
    # plt.tight_layout()
    # fig.savefig('cnn_reward_histogram.png')
    # plt.show()

    # # 3) Boxplot of rewards
    # fig, ax = plt.subplots(figsize=(4, 6), dpi=dpi)
    # ax.boxplot(rewards, vert=True, patch_artist=True,
    #            boxprops=dict(facecolor='white', edgecolor='black'),
    #            medianprops=dict(color='red', linewidth=2))
    # ax.set_ylabel('Episode Total Reward', fontsize=label_fs)
    # ax.set_title('CNN-DQN Reward Boxplot', fontsize=title_fs)
    # ax.grid(axis='y', **grid_kw)
    # ax.tick_params(labelsize=tick_fs)
    # plt.tight_layout()
    # fig.savefig('cnn_reward_boxplot.png')
    # plt.show()

    # 4) Steps curve
    low_s, high_s = np.percentile(steps, [1, 99])
    fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
    ax.plot(steps, alpha=0.3, linewidth=1, label='Episode steps')
    # ax.plot(ma_s, linewidth=line_lw, label=f'{ma_window}-ep MA')
    ax.axhline(mean_s, color='red', linestyle='--', linewidth=1,
               label=f'Overall mean = {mean_s:.2f}')
    ax.set_ylim(low_s, high_s)
    ax.set_xlabel('Episode', fontsize=label_fs)
    ax.set_ylabel('Steps', fontsize=label_fs)
    ax.set_title('CNN-DQN Evaluation: Steps per Episode', fontsize=title_fs)
    ax.legend(fontsize=label_fs)
    ax.grid(**grid_kw)
    ax.tick_params(labelsize=tick_fs)
    plt.tight_layout()
    fig.savefig('cnn_steps_curve.png')
    plt.show()


if __name__ == '__main__':
    model_file = 'cnn_dqn_model_20250404-234200.pth'
    evaluate_and_plot(model_path=model_file)
