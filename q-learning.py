import argparse
import hashlib
import pickle
import random
import numpy as np
from vis_gym import *


class QLearningAgent:
    def __init__(self, env=None, grid_size=5, gamma=0.9,
                 epsilon=1.0, decay_rate=0.999, gui_flag=False):
        # Environment and visualization
        setup(GUI=gui_flag)
        self.env = env or game
        self.grid_size = grid_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.gui_flag = gui_flag

        # Q-learning tables
        self.Q_table = {}
        self.update_counts = {}

    def get_partial_observation(self, player_position, wall_positions):
        """
        Extract a 3x3 grid around the player's position.
        Walls as 1, free as 0, out-of-bounds as 1.
        """
        x, y = player_position
        local = []
        for i in range(x - 1, x + 2):
            row = []
            for j in range(y - 1, y + 2):
                if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                    row.append(1 if (i, j) in wall_positions else 0)
                else:
                    row.append(1)
            local.append(tuple(row))
        return tuple(local)

    def compute_state_hash(self, player_position, wall_positions):
        """
        SHA256 hash of the 3x3 local observation and goal room.
        """
        partial = self.get_partial_observation(player_position, wall_positions)
        rep = (partial, self.env.goal_room)
        return hashlib.sha256(str(rep).encode('utf-8')).hexdigest()

    def initialize_state(self, state_hash):
        """
        Ensure Q_table and update_counts have entries for the state.
        """
        if state_hash not in self.Q_table:
            self.Q_table[state_hash] = np.zeros(len(self.env.actions))
            self.update_counts[state_hash] = np.zeros(len(self.env.actions))

    def choose_action(self, state_hash):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.epsilon or self.Q_table[state_hash].sum() == 0:
            return random.choice(self.env.actions)
        idx = np.argmax(self.Q_table[state_hash])
        return self.env.actions[idx]

    def update_q(self, prev_hash, action_idx, reward, new_hash):
        """
        Update Q-value for a given state-action pair.
        """
        self.initialize_state(prev_hash)
        self.initialize_state(new_hash)

        count = self.update_counts[prev_hash][action_idx]
        alpha = 1.0 / (1.0 + count)
        best_future = np.max(self.Q_table[new_hash])
        old_q = self.Q_table[prev_hash][action_idx]

        # Q-learning update
        new_q = (1 - alpha) * old_q + alpha * (reward + self.gamma * best_future)
        self.Q_table[prev_hash][action_idx] = new_q
        self.update_counts[prev_hash][action_idx] += 1

    def train(self, num_episodes=200000):
        """
        Run Q-learning for a number of episodes.
        Returns:
            Q_table
        """
        for ep in range(num_episodes):
            obs, reward, done, info = self.env.reset()
            total_reward = 0
            steps = 0

            while not done:
                prev_hash = self.compute_state_hash(
                    self.env.current_state['player_position'],
                    self.env.wall_positions)

                self.initialize_state(prev_hash)
                action = self.choose_action(prev_hash)

                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1

                new_hash = self.compute_state_hash(
                    self.env.current_state['player_position'],
                    self.env.wall_positions)

                action_idx = self.env.actions.index(info.get('action'))
                self.update_q(prev_hash, action_idx, reward, new_hash)

                if self.gui_flag:
                    refresh(obs, reward, done, info)

            # Decay epsilon
            self.epsilon *= self.decay_rate
            if ep % 1000 == 0:
                print(f"Episode {ep}: steps={steps}, reward={total_reward:.2f}, epsilon={self.epsilon:.4f}")

        return self.Q_table

    def save(self, filename='Q_table.pickle'):
        """
        Save the learned Q-table to disk.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.Q_table, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename='Q_table.pickle'):
        """
        Load a Q-table from disk.
        """
        with open(filename, 'rb') as f:
            self.Q_table = pickle.load(f)

    def play(self, episodes=1, max_steps_per_episode=50):
        """
        Run episodes using the learned policy.
        """
        total_reward = 0
        for ep in range(episodes):
            obs, reward, done, info = self.env.reset()
            steps = 0
            while not done and steps < max_steps_per_episode:
                state_hash = self.compute_state_hash(
                    self.env.current_state['player_position'],
                    self.env.wall_positions)
                action_idx = np.argmax(self.Q_table.get(state_hash, np.zeros(len(self.env.actions))))
                action = self.env.actions[action_idx]
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
                if self.gui_flag:
                    refresh(obs, reward, done, info)
            print(f"Play Episode {ep + 1}: steps={steps}, reward={reward:.2f}")

        avg = total_reward / episodes
        print(f"Average reward over {episodes} episodes: {avg:.2f}")
        return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play with QLearningAgent.")
    parser.add_argument('--mode', choices=['train', 'play'], default='play',
                        help="Mode: 'train' to train the agent, 'play' to run with a saved Q-table")
    parser.add_argument('--episodes', type=int, default=1000000,
                        help="Number of episodes for training or playing")
    parser.add_argument('--gui', action='store_true', help="Enable GUI visualization")
    parser.add_argument('--decay', type=float, default=0.999999,
                        help="Epsilon decay rate")
    args = parser.parse_args()

    agent = QLearningAgent(
        gui_flag=args.gui,
        decay_rate=args.decay
    )

    if args.mode == 'train':
        agent.train(num_episodes=args.episodes)
        agent.save('Q_table.pickle')
    else:
        agent.load('Q_table.pickle')
        agent.play(episodes=args.episodes)
