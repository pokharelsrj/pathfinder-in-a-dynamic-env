import hashlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
from vis_gym import *

# --- ENVIRONMENT SETUP ---
gui_flag = False
setup(GUI=gui_flag)
env = game

# --- OBSERVATION UTILITIES ---
def get_partial_observation(player_position, wall_positions, grid_size=8):
    x, y = player_position
    grid = []
    for i in range(x - 1, x + 2):
        row = []
        for j in range(y - 1, y + 2):
            if 0 <= i < grid_size and 0 <= j < grid_size:
                row.append(1 if (i, j) in wall_positions else 0)
            else:
                row.append(1)
        grid.append(tuple(row))
    return tuple(grid)

def compute_partial_state_hash(player_position, wall_positions, grid_size):
    partial_obs = get_partial_observation(player_position, wall_positions, grid_size)
    state_representation = (partial_obs, env.goal_room)
    state_str = str(state_representation)
    return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

# --- EVALUATION & PLOTTING ---
def evaluate_and_plot(q_table_path='Q_table.pickle',
                      num_episodes=2000,
                      ma_window=100):
    # Load Q‑table
    with open(q_table_path, 'rb') as f:
        Q_table = pickle.load(f)

    rewards = []
    for ep in range(1, num_episodes + 1):
        obs, _, done, _ = env.reset()
        ep_reward = 0
        while not done:
            st = compute_partial_state_hash(
                env.current_state['player_position'],
                env.wall_positions,
                env.grid_size
            )
            idx = np.argmax(Q_table.get(st, np.zeros(len(env.actions))))
            action = env.actions[idx]
            obs, r, done, _ = env.step(action)
            ep_reward += r
            if gui_flag:
                refresh(obs, r, done, _)
        rewards.append(ep_reward)

        # occasional logging
        if ep % 500 == 0 or ep == 1:
            ma = np.mean(rewards[-ma_window:])
            print(f"Episode {ep:4d}  Reward {ep_reward:6.2f}  {ma_window}-ep moving avg {ma:6.2f}")

    env.close()

    # Convert to array & compute stats
    rewards = np.array(rewards)
    mean_r, std_r, med_r = rewards.mean(), rewards.std(), np.median(rewards)
    print(f"\nOverall (n={num_episodes}): μ={mean_r:.2f}, σ={std_r:.2f}, med={med_r:.2f}")

    # Moving average (for curve)
    if len(rewards) >= ma_window:
        window = np.ones(ma_window) / ma_window
        ma = np.convolve(rewards, window, mode='valid')
        ma = np.concatenate([np.full(ma_window - 1, np.nan), ma])
    else:
        ma = np.full_like(rewards, np.nan)

    # Common styling kwargs
    label_fs = 14
    title_fs = 16
    tick_fs = 12
    line_lw = 2
    grid_kw = dict(linestyle='--', alpha=0.5)

    # 1) Reward curve + moving average + mean line, cropped to 1st–99th percentiles
    low, high = np.percentile(rewards, [1, 99])
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.plot(rewards, alpha=0.3, linewidth=1, label='Episode reward')
    ax.plot(ma, linewidth=line_lw, label=f'{ma_window}-episode moving average')
    ax.axhline(mean_r, linestyle='--', linewidth=1, label=f'Overall mean = {mean_r:.2f}')
    ax.set_ylim(low, high)
    ax.set_xlabel('Episode', fontsize=label_fs)
    ax.set_ylabel('Total Reward', fontsize=label_fs)
    ax.set_title(
        f'Agent Performance Over Episodes\n(shown between {low:.0f} and {high:.0f})',
        fontsize=title_fs
    )
    ax.legend(fontsize=label_fs)
    ax.grid(**grid_kw)
    ax.tick_params(axis='both', labelsize=tick_fs)
    plt.tight_layout()
    fig.savefig('reward_curve.png')
    plt.show()

    # 2) Histogram of rewards (clamp top 1%)
    upper = np.percentile(rewards, 99)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.hist(
        np.clip(rewards, rewards.min(), upper),
        bins=30,
        alpha=0.7,
        edgecolor='black'
    )
    ax.set_xlabel('Episode Total Reward', fontsize=label_fs)
    ax.set_ylabel('Frequency', fontsize=label_fs)
    ax.set_title('Distribution of Episode Rewards (clamped at 99th pct)', fontsize=title_fs)
    ax.grid(**grid_kw)
    ax.tick_params(labelsize=tick_fs)
    plt.tight_layout()
    fig.savefig('reward_histogram.png')
    plt.show()

    # 3) Boxplot of rewards (no fliers)
    fig, ax = plt.subplots(figsize=(4, 6), dpi=300)
    ax.boxplot(
        rewards,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor='white', edgecolor='black'),
        medianprops=dict(color='red', linewidth=2),
        showfliers=False
    )
    ax.set_ylabel('Episode Total Reward', fontsize=label_fs)
    ax.set_title('Boxplot of Episode Rewards (no fliers)', fontsize=title_fs)
    ax.grid(axis='y', **grid_kw)
    ax.tick_params(labelsize=tick_fs)
    plt.tight_layout()
    fig.savefig('reward_boxplot.png')
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()
