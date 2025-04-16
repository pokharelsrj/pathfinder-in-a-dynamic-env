import argparse
import random
import time
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from vis_gym import *  # provides setup, game, refresh

# ================ TRANSITION ================
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# ================ CNN-DQN MODEL ================
class CNNDQN(nn.Module):
    def __init__(self, input_channels, h, w, outputs):
        super(CNNDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(16)

        # compute flattened size: input 8x8 -> conv1->7x7->pool->6x6->conv2->5x5
        linear_input_size = 16 * 5 * 5

        self.fc1 = nn.Linear(linear_input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, outputs)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ================ REPLAY MEMORY ================
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ================ AGENT ================
class CNNDQNAgent:
    def __init__(
        self,
        gui_flag,
        device,
        gamma,
        epsilon_start,
        epsilon_min,
        epsilon_decay,
        tau,
        lr,
        batch_size,
        replay_size,
    ):
        # setup environment
        setup(GUI=gui_flag)
        self.env = game
        self.gui_flag = gui_flag
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.lr = lr
        self.batch_size = batch_size

        self.n_actions = len(self.env.actions)
        self.memory = ReplayMemory(replay_size)

        # networks
        self.policy_net = CNNDQN(3, self.env.grid_size, self.env.grid_size, self.n_actions).to(device)
        self.target_net = CNNDQN(3, self.env.grid_size, self.env.grid_size, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

    def process_observation(self):
        G = self.env.grid_size
        walls = np.zeros((G, G), dtype=np.float32)
        agent = np.zeros((G, G), dtype=np.float32)
        goal = np.zeros((G, G), dtype=np.float32)
        for (i, j) in self.env.wall_positions:
            walls[i, j] = 1.0
        pi, pj = self.env.current_state['player_position']
        agent[pi, pj] = 1.0
        gi, gj = self.env.goal_room
        goal[gi, gj] = 1.0
        return np.stack([walls, agent, goal], axis=0)

    def get_state_tensor(self):
        grid = self.process_observation()
        t = torch.tensor(grid, dtype=torch.float32, device=self.device).unsqueeze(0)
        return t

    def select_action(self, state):
        if random.random() < self.epsilon:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long), True
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1), False

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state], device=self.device, dtype=torch.bool
        )
        non_final_next = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action = self.policy_net(state_batch).gather(1, action_batch)
        next_vals = torch.zeros(self.batch_size, device=self.device)
        if non_final_mask.sum() > 0:
            with torch.no_grad():
                next_actions = self.policy_net(non_final_next).max(1)[1].unsqueeze(1)
                next_vals[non_final_mask] = self.target_net(non_final_next).gather(1, next_actions).squeeze(1)
        expected = reward_batch + self.gamma * next_vals
        loss = F.mse_loss(state_action, expected.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def soft_update(self):
        for t, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            t.data.copy_(t.data * (1.0 - self.tau) + p.data * self.tau)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, episodes, log_dir):
        writer = SummaryWriter(log_dir)
        total_loss = 0.0
        steps = 0
        for ep in range(episodes):
            obs, reward, done, _ = self.env.reset()
            state = self.get_state_tensor()
            ep_reward, ep_steps = 0, 0
            writer.add_scalar('Epsilon', self.epsilon, ep)
            while not done:
                action, _ = self.select_action(state)
                obs, reward, done, info = self.env.step(action)
                next_state = None if done else self.get_state_tensor()
                self.memory.push(state, action, next_state, torch.tensor([reward], device=self.device))
                state = next_state

                loss = self.optimize_model()
                if loss is not None:
                    total_loss += loss
                    steps += 1
                    writer.add_scalar('Loss', total_loss / steps, steps)
                self.soft_update()
                ep_reward += reward
                ep_steps += 1
                if self.gui_flag:
                    refresh(obs, reward, done, info)

            writer.add_scalar('Reward', ep_reward, ep)
            writer.add_scalar('Steps', ep_steps, ep)
            self.decay_epsilon()
            print(f"Episode {ep+1}/{episodes}, Steps: {ep_steps}, Reward: {ep_reward}, Epsilon: {self.epsilon:.4f}")
        writer.close()
        model_file = f"cnn_dqn_{int(time.time())}.pth"
        torch.save(self.policy_net.state_dict(), model_file)
        print(f"Model saved to {model_file}")

    def evaluate(self, episodes, model_path=None):
        if model_path:
            self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()
        for ep in range(episodes):
            obs, reward, done, _ = self.env.reset()
            state = self.get_state_tensor()
            total_reward = 0
            while not done:
                with torch.no_grad():
                    action = self.policy_net(state).max(1)[1].view(1,1)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                if self.gui_flag:
                    refresh(obs, reward, done, info)
                state = None if done else self.get_state_tensor()
            print(f"Eval {ep+1}/{episodes}, Reward: {total_reward}")


# ================ MAIN ================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate a CNN-DQN agent on grid world.")
    parser.add_argument('--mode', choices=['train', 'eval'], default='train', help="Mode: train or eval")
    parser.add_argument('--episodes', type=int, default=3000, help="Number of episodes")
    parser.add_argument('--gui', action='store_true', help="Enable GUI visualization")
    parser.add_argument('--gamma', type=float, default=0.9, help="Discount factor")
    parser.add_argument('--epsilon_start', type=float, default=1.0, help="Starting epsilon")
    parser.add_argument('--epsilon_min', type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument('--epsilon_decay', type=float, default=0.9995, help="Epsilon decay rate")
    parser.add_argument('--tau', type=float, default=0.005, help="Target network soft update parameter")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--replay_size', type=int, default=10000, help="Replay memory size")
    parser.add_argument('--log_dir', type=str, default=None, help="Tensorboard log directory (train mode)")
    parser.add_argument('--model_path', type=str, default=None, help="Path to saved model (eval mode)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    agent = CNNDQNAgent(
        gui_flag=args.gui,
        device=device,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        tau=args.tau,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
    )

    if args.mode == 'train':
        log_dir = args.log_dir or f"runs/cnn_dqn_{int(time.time())}"
        agent.train(args.episodes, log_dir)
    else:
        agent.evaluate(args.episodes, args.model_path)
