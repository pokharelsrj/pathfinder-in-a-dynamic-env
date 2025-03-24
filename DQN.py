import math
from collections import namedtuple, deque
from itertools import count

from torch import nn, optim
import torch.nn.functional as F
import torch
from vis_gym import *

# Set up environment.
gui_flag = True
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
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Environment info.
n_actions = len(env.actions)
n_observations = 9

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
        self.layer_1 = nn.Linear(n_observations, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, n_actions)

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
    return tuple(flattened_grid)


# Helper function: Epsilon-greedy action selection.
def select_action(state, policy_net, steps_done, env):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            action = policy_net(state).max(1).indices.view(1, 1)
    else:
        action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    return action, steps_done


# Helper function: Optimize the model using a batch of transitions.
def optimize_model(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# Helper function: Soft update the target network.
def soft_update(target_net, policy_net, tau):
    target_dict = target_net.state_dict()
    policy_dict = policy_net.state_dict()
    for key in policy_dict:
        target_dict[key] = policy_dict[key] * tau + target_dict[key] * (1 - tau)
    target_net.load_state_dict(target_dict)


# Training loop.
def train_agent(env, policy_net, target_net, optimizer, memory, num_episodes):
    steps_done = 0
    for ep in range(num_episodes):
        obs, reward, done, info = env.reset()
        flat_obs = get_flattened_observation(env)
        state = torch.tensor(flat_obs, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0

        for t in count():
            action, steps_done = select_action(state, policy_net, steps_done, env)
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

            optimize_model(policy_net, target_net, optimizer, memory)
            soft_update(target_net, policy_net, TAU)

            if done:
                print(f"Episode {ep + 1} finished after {t + 1} steps, Total Reward: {episode_reward}")
                break

    torch.save(policy_net.state_dict(), 'dqn_model.pth')
    print("Training complete. Model saved as dqn_model.pth.")


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

    if TRAIN_MODE:
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            num_episodes = 600
        else:
            num_episodes = 50
        train_agent(env, policy_net, target_net, optimizer, memory, num_episodes)
    else:
        # Load the saved model weights if available.
        policy_net.load_state_dict(torch.load('dqn_model.pth', map_location=device))
        evaluate_agent(env, policy_net, num_eval_episodes=100)
