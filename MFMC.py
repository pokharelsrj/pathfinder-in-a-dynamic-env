import hashlib
import random
import time
import pickle
import numpy as np
from vis_gym import *

gui_flag = False  # Set to True to enable the game state visualization
setup(GUI=gui_flag)
env = game  # Gym environment already initialized within vis_gym.py


# env.render()  # Uncomment to print game state info


# def hash(obs):
#     x, y = obs['player_position']
#     h = obs['player_health']
#     g = obs['wall_in_cell']
#     if not g:
#         g = 0
#     else:
#         g = int(g[-1])
#
#     return x * (5 * 2 * 5) + y * (2 * 5) + h * 5 + g

def get_partial_observation(player_position, wall_positions, grid_size=8):
    """
    Extract a 5x5 grid around the player's position.
    Walls are indicated with 1, free cells with 0.
    Out-of-bound cells are marked with -1.

    Args:
        player_position (tuple): (x, y) coordinates of the player.
        wall_positions (list or tuple): List of wall coordinate tuples.
        grid_size (int): Size of the full grid (default=5).

    Returns:
        tuple: A tuple of 5 tuples (each of length 5) representing the local grid.
    """
    x, y = player_position
    local_grid = []
    for i in range(x - 1, x + 2):  # actually generates: x-1, x, x+1
        row = []
        for j in range(y - 1, y + 2):  # actually generates: y-1, y, y+1
            if 0 <= i < grid_size and 0 <= j < grid_size:
                row.append(1 if (i, j) in wall_positions else 0)
            else:
                row.append(1)  # note: out-of-bound cells are marked with 1 instead of -1
        local_grid.append(tuple(row))
    return tuple(local_grid)


get_partial_observation((0, 0), [])


def compute_partial_state_hash(player_position, wall_positions, grid_size=5):
    """
    Computes a stable hash for the 3x3 observation state around the agent.

    Args:
        player_position (tuple): The player's (x, y) position.
        wall_positions (list or tuple): List of wall positions.
        grid_size (int): Size of the full grid.

    Returns:
        str: A SHA256 hash hex digest representing the 3x3 observation state.
    """
    partial_obs = get_partial_observation(player_position, wall_positions, grid_size)
    # Create a string representation that preserves the layout.
    state_str = str(partial_obs)
    return hashlib.sha256(state_str.encode('utf-8')).hexdigest()


# def random_movement():
#     i = 0
#     total_reward = 0
#     while i != 1:
#         obs, reward, done, info = env.reset()
#         while not done:
#             action = random.choice(env.actions)
#             obs, reward, done, info = env.step(action)
#             total_reward += reward
#             if gui_flag:
#                 refresh(obs, reward, done, info)  # Update the game screen [GUI only]
#         i += 1
#
#     print(total_reward)
#     env.close()
#
#
# random_movement()


# def Q_learning(num_episodes=200, gamma=0.9, epsilon=1, decay_rate=0.999):
#     """
#     Run Q-learning algorithm for a specified number of episodes.
#
#     Parameters:
#     - num_episodes (int): Number of episodes to run.
#     - gamma (float): Discount factor.
#     - epsilon (float): Exploration rate.
#     - decay_rate (float): Rate at which epsilon decays. Epsilon is decayed as epsilon = epsilon * decay_rate after each episode.
#
#     Returns:
#     - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
#     """
#     Q_table = {}
#     number_of_updates = {}
#
#     current_ep = 0
#     while current_ep != num_episodes:
#         obs, reward, done, info = env.reset()
#         total_rewards = 0
#         no_of_moves = 0
#         while not done:
#             prev_state = compute_partial_state_hash(env.current_state['player_position'],
#                                                     env.wall_positions,
#                                                     env.grid_size)
#             if Q_table.get(prev_state) is None or random.random() < epsilon:
#                 action_to_take = random.choice(env.actions)
#             else:
#                 action_idx = np.argmax(Q_table[prev_state])
#                 action_to_take = env.actions[action_idx]
#
#             obs, reward, done, info = env.step(action_to_take)
#             total_rewards += reward
#             state = compute_partial_state_hash(env.current_state['player_position'],
#                                                env.wall_positions,
#                                                env.grid_size)
#             if gui_flag:
#                 refresh(obs, reward, done, info)  # Update the game screen [GUI only]
#
#             action = info.get('action')
#             index = env.actions.index(action)
#
#             if Q_table.get(state) is None:
#                 Q_table.update({state: np.zeros(len(env.actions))})
#
#             if Q_table.get(prev_state) is None:
#                 Q_table.update({prev_state: np.zeros(len(env.actions))})
#
#             if number_of_updates.get(state) is None:
#                 number_of_updates.update({state: np.zeros(len(env.actions))})
#
#             if number_of_updates.get(prev_state) is None:
#                 number_of_updates.update({prev_state: np.zeros(len(env.actions))})
#
#             Q_opt_prev = Q_table[prev_state][index]
#             number_of_updates_prev = number_of_updates[prev_state][index]
#
#             eta = 1 / (1 + number_of_updates_prev)
#             V_opt = np.max(Q_table[state])
#
#             Q_opt_curr = ((1 - eta) * Q_opt_prev) + eta * (reward + (gamma * V_opt))
#
#             Q_table[prev_state][index] = Q_opt_curr
#             number_of_updates[prev_state][index] += 1
#             no_of_moves += 1
#
#         print(current_ep, no_of_moves, total_rewards, epsilon)
#
#         current_ep = current_ep + 1
#         epsilon = epsilon * decay_rate
#
#     return Q_table
#
#
# decay_rate = 0.999999
#
# Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate)  # Run Q-learning
#
# # Save the Q-table dict to a file
# with open('Q_table.pickle', 'wb') as handle:
#     pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
Uncomment the code below to play an episode using the saved Q-table. Useful for debugging/visualization.

Comment before final submission or autograder may fail.
'''

Q_table = np.load('Q_table.pickle', allow_pickle=True)
print(len(Q_table))
i = 0
total_reward = 0
total_steps = 0
while i != 5000:
    obs, reward, done, info = env.reset()
    while not done:
        state = compute_partial_state_hash(env.current_state['player_position'],
                                           env.wall_positions,
                                           env.grid_size)
        action_index = np.argmax(Q_table[state])
        action = env.actions[action_index]
        obs, reward, done, info = env.step(action)
        total_reward += reward
        total_steps += 1
        if gui_flag:
            refresh(obs, reward, done, info)  # Update the game screen [GUI only]
    i += 1
    print(i)

print(total_reward / 5000)
print(total_steps / 5000)

# Close the
env.close()  # Close the environment
