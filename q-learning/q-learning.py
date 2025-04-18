import argparse
import hashlib
import os
import pickle
import numpy as np
from env.gui import *

# This code is part of the Q-learning agent approach that we initially tried (not recommended).
# For our environment with visual state spaces (images/grids), we wouldn't use this because the Q-table would become impossibly large and sparse.
# Use the trained model included in this directory based on your use-case, i.e. random goal or fixed goal.
class QLearningAgent:
    def __init__(self, gamma=0.9, epsilon=1.0,
                 decay_rate=0.999999, gui_flag=False, fixed_goal=False):
        self.gui = DynamicMazeGUI(gui_flag, fixed_goal)
        self.gui.setup()

        self.env = self.gui.game
        self.gui_flag = gui_flag
        self.grid_size = self.env.grid_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate

        self.Q_table = {}
        self.number_of_updates = {}

    def get_partial_observation(self, player_position, wall_positions):
        """
        Extract a 3x3 grid around the player's position.
        Walls as 1, free as 0, out‐of‐bounds as 1.
        """
        x, y = player_position
        local_grid = []
        for i in range(x - 1, x + 2):
            row = []
            for j in range(y - 1, y + 2):
                if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                    row.append(1 if (i, j) in wall_positions else 0)
                else:
                    row.append(1)
            local_grid.append(tuple(row))
        return tuple(local_grid)

    def compute_partial_state_hash(self, player_position, wall_positions):
        """
        Computes a stable SHA256 hash for the 3x3 observation state.
        """
        partial_obs = self.get_partial_observation(player_position, wall_positions)
        state_representation = (partial_obs, self.env.goal_room)
        return hashlib.sha256(str(state_representation).encode('utf-8')).hexdigest()

    def train(self, num_episodes=1000000):
        """
        Run Q‑learning for a specified number of episodes.
        """
        for ep in range(num_episodes):
            obs, reward, done, info = self.env.reset()
            total_rewards = 0
            no_of_moves = 0

            while not done:
                prev_state = self.compute_partial_state_hash(
                    self.env.current_state['player_position'],
                    self.env.wall_positions
                )

                # Epsilon‐greedy action selection
                if self.Q_table.get(prev_state) is None or random.random() < self.epsilon:
                    action_to_take = random.choice(self.env.actions)
                else:
                    idx = np.argmax(self.Q_table[prev_state])
                    action_to_take = self.env.actions[idx]

                obs, reward, done, info = self.env.step(action_to_take)
                total_rewards += reward

                state = self.compute_partial_state_hash(
                    self.env.current_state['player_position'],
                    self.env.wall_positions
                )

                if self.gui_flag:
                    self.gui.refresh(obs, reward, done, info)

                action = info.get('action')
                index = self.env.actions.index(action)

                # Initialize missing entries
                if self.Q_table.get(state) is None:
                    self.Q_table[state] = np.zeros(len(self.env.actions))

                if self.Q_table.get(prev_state) is None:
                    self.Q_table[prev_state] = np.zeros(len(self.env.actions))

                if self.number_of_updates.get(state) is None:
                    self.number_of_updates[state] = np.zeros(len(self.env.actions))

                if self.number_of_updates.get(prev_state) is None:
                    self.number_of_updates[prev_state] = np.zeros(len(self.env.actions))

                # Q‐learning update
                q_opt_prev = self.Q_table[prev_state][index]
                updates_prev = self.number_of_updates[prev_state][index]
                eta = 1.0 / (1 + updates_prev)
                v_opt = np.max(self.Q_table[state])
                q_opt_curr = ((1 - eta) * q_opt_prev) + eta * (reward + self.gamma * v_opt)

                self.Q_table[prev_state][index] = q_opt_curr
                self.number_of_updates[prev_state][index] += 1

                no_of_moves += 1

            # End of episode
            print(ep, no_of_moves, total_rewards, self.epsilon)
            self.epsilon *= self.decay_rate

        return self.Q_table

    def save(self, filename='Q_table_random_goal.pickle'):
        path = os.path.join(os.path.dirname(__file__), "trained_model", filename)
        with open(path, 'wb') as handle:
            pickle.dump(self.Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename='Q_table_random_goal.pickle'):
        path = os.path.join(os.path.dirname(__file__), "trained_model", filename)
        with open(path, 'rb') as handle:
            self.Q_table = pickle.load(handle)

    def play(self, episodes=5000):
        """
        Run episodes using the learned Q‑table.
        """
        total_reward = 0
        total_step = 0

        for i in range(episodes):
            obs, reward, done, info = self.env.reset()
            while not done:
                state = self.compute_partial_state_hash(
                    self.env.current_state['player_position'],
                    self.env.wall_positions
                )
                action_index = np.argmax(self.Q_table.get(state, np.zeros(len(self.env.actions))))
                action = self.env.actions[action_index]
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                total_step += 1
                if self.gui_flag:
                    self.gui.refresh(obs, reward, done, info)

        print(total_reward / episodes)
        print(total_step / episodes)
        return total_reward / episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play with QLearningAgent.")
    parser.add_argument('--mode', choices=['train', 'play'], default='play',
                        help="Mode: 'train' to train, 'play' to run with a saved Q-table")
    parser.add_argument('--episodes', type=int, default=1000000,
                        help="Number of episodes for training or playing")
    parser.add_argument('--gui', action='store_true', help="Enable GUI visualization")
    parser.add_argument('--gamma', type=float, default=0.9,
                        help="Discount factor for Q-learning")
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help="Initial exploration rate")
    parser.add_argument('--decay_rate', type=float, default=0.999999,
                        help="Epsilon decay rate per episode")
    parser.add_argument('--qtable', type=str,
                        help="Path to save/load the Q-table")
    parser.add_argument('--fixed_goal', action='store_true')

    args = parser.parse_args()
    agent = QLearningAgent(
        gamma=args.gamma,
        epsilon=args.epsilon,
        decay_rate=args.decay_rate,
        gui_flag=args.gui,
        fixed_goal=args.fixed_goal
    )

    if args.mode == 'train':
        agent.train(num_episodes=args.episodes)
        agent.save(args.qtable)
    else:
        agent.load(args.qtable)
        agent.play(episodes=args.episodes)
