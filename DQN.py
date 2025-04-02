import math
import random
import time
from collections import namedtuple, deque
from itertools import count

from torch import nn, optim
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from vis_gym import *

# Set up environment.
gui_flag = False
setup(GUI=gui_flag)
env = game

# Device selection.
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print("Using device:", device)

# Hyperparameters and constants.
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 3000
TAU = 0.005
LR = 1e-4
EPSILON = 1.0

# Environment info.
n_actions = len(env.actions)
n_observations = 10

# Transition tuple for replay memory.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# Replay Memory class.
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN model definition.
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer_1 = nn.Linear(n_observations, 25)
        self.layer_2 = nn.Linear(25, 15)
        self.layer_3 = nn.Linear(15, n_actions)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)


# Initialize networks, optimizer, and memory.
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


def cell_hash(cell):
    i, j = cell
    return i * 9 + j


# Helper function: Process observation into a flattened 3x3 grid.
def get_flattened_observation(env):
    player_position = env.current_state['player_position']
    wall_positions = env.wall_positions
    grid_size = env.grid_size
    x, y = player_position
    flattened_grid = []
    for i in range(x - 1, x + 2):  # x-1, x, x+1
        for j in range(y - 1, y + 2):  # y-1, y, y+1
            if 0 <= i < grid_size and 0 <= j < grid_size:
                flattened_grid.append(1 if (i, j) in wall_positions else 0)
            else:
                flattened_grid.append(1)  # out-of-bound cells marked with 1
    flattened_grid.append(cell_hash(env.goal_room))
    return tuple(flattened_grid)


# Modified epsilon-greedy action selection that returns a flag for random actions.
# def select_action(state, policy_net, steps_done, env):
#     sample = random.random()
#     eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
#     steps_done += 1
#     if sample > eps_threshold:
#         with torch.no_grad():
#             action = policy_net(state).max(1).indices.view(1, 1)
#         is_random = False
#     else:
#         action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
#         is_random = True
#     return action, steps_done, eps_threshold, is_random

def select_action(state, policy_net, env, epsilon):
    if random.random() < epsilon:
        action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        is_random = True
    else:
        action = policy_net(state).max(1).indices.view(1, 1)
        is_random = False
    return action, is_random


# Modified optimize_model that logs Q-value stats and gradient norms.
def optimize_model(policy_net, target_net, optimizer, memory, global_step, writer):
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Get Q-values for the current states.
    q_values = policy_net(state_batch)
    state_action_values = q_values.gather(1, action_batch)

    # Log Q-value statistics.
    writer.add_scalar("QValues/Mean", q_values.mean().item(), global_step)
    writer.add_scalar("QValues/Max", q_values.max().item(), global_step)
    writer.add_scalar("QValues/Std", q_values.std().item(), global_step)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    # Compute and log the total gradient norm.
    total_grad_norm = 0.0
    for p in policy_net.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    writer.add_scalar("Gradient/Norm", total_grad_norm, global_step)

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()


# Soft update of the target network.
def soft_update(target_net, policy_net, tau):
    target_dict = target_net.state_dict()
    policy_dict = policy_net.state_dict()
    for key in policy_dict:
        target_dict[key] = policy_dict[key] * tau + target_dict[key] * (1 - tau)
    target_net.load_state_dict(target_dict)


# Training loop with additional metric logging.
def train_agent(env, policy_net, target_net, optimizer, memory, num_episodes, writer):
    global EPSILON  # Use the global EPSILON variable.
    loss_history = deque(maxlen=100)

    for ep in range(num_episodes):
        episode_start_time = time.time()
        obs, reward, done, info = env.reset()
        flat_obs = get_flattened_observation(env)
        state = torch.tensor(flat_obs, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0
        episode_random_count = 0
        episode_greedy_count = 0
        episode_action_counts = {i: 0 for i in range(n_actions)}

        # Optionally, you can log the current epsilon for the episode.
        writer.add_scalar("Epsilon/episode", EPSILON, ep)

        for t in count():
            action, is_random = select_action(state, policy_net, env, EPSILON)
            # Update action counts.
            if is_random:
                episode_random_count += 1
            else:
                episode_greedy_count += 1
            episode_action_counts[action.item()] += 1

            obs, reward, done, info = env.step(action)
            reward_tensor = torch.tensor([reward], device=device)
            episode_reward += reward

            if done:
                next_state = None
            else:
                flat_obs = get_flattened_observation(env)
                next_state = torch.tensor(flat_obs, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward_tensor)
            state = next_state

            loss_value = optimize_model(policy_net, target_net, optimizer, memory, t, writer)
            if loss_value is not None:
                writer.add_scalar("Loss/train", loss_value, t)
                loss_history.append(loss_value)
                moving_avg_loss = sum(loss_history) / len(loss_history)
                writer.add_scalar("Loss/MovingAverage", moving_avg_loss, t)

            soft_update(target_net, policy_net, TAU)

            if done:
                episode_duration = time.time() - episode_start_time
                writer.add_scalar("Reward/episode", episode_reward, ep)
                writer.add_scalar("Episode/Length", t + 1, ep)
                writer.add_scalar("Episode/Duration", episode_duration, ep)
                writer.add_scalar("ReplayMemory/Size", len(memory), ep)

                print(
                    f"Episode {ep + 1} finished after {t + 1} steps, Total Reward: {episode_reward}, Duration: {episode_duration:.2f}s")
                break

        # Decay epsilon after each episode.
        EPSILON *= 0.9997

    # Save model with timestamp.
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f'dqn_model_{timestamp}.pth'
    torch.save(policy_net.state_dict(), model_filename)
    print(f"Training complete. Model saved as {model_filename}.")


# Evaluation loop.
def evaluate_agent(env, policy_net, num_eval_episodes=5):
    policy_net.eval()
    for episode in range(num_eval_episodes):
        obs, reward, done, info = env.reset()
        state = torch.tensor(get_flattened_observation(env), dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0

        while True:
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if gui_flag:
                refresh(obs, reward, done, info)

            if done:
                print(f"Evaluation Episode {episode + 1}: Total Reward = {total_reward}")
                break
            state = torch.tensor(get_flattened_observation(env), dtype=torch.float32, device=device).unsqueeze(0)


if __name__ == "__main__":
    # Set this flag to True to train the agent, or False to evaluate a saved policy.
    TRAIN_MODE = False

    # Create a log directory with timestamp appended.
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/dqn_experiment_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)

    if TRAIN_MODE:
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            num_episodes = 5000
        else:
            num_episodes = 50
        train_agent(env, policy_net, target_net, optimizer, memory, num_episodes, writer)
        writer.close()
    else:
        # Load the saved model weights if available.
        policy_net.load_state_dict(torch.load('dqn_model_20250402-145624.pth', map_location=device))
        evaluate_agent(env, policy_net, num_eval_episodes=100)
