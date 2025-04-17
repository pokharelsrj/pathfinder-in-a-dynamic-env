import argparse
import random
import time
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
from CDQN import CDQN
from ReplayMemory import ReplayMemory
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from vis_gym import setup, game, refresh


class CDQNAgent:
    """CNN-DQN Agent with train and play methods, hyperparams injected."""

    def __init__(
            self,
            gui_flag: bool,
            device: torch.device,
            gamma: float,
            epsilon_start: float,
            epsilon_min: float,
            epsilon_decay: float,
            tau: float,
            lr: float,
            batch_size: int,
            replay_size: int,
    ):
        # env setup
        setup(GUI=gui_flag)
        self.env = game

        # device & exploration params
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau

        # training params
        self.batch_size = batch_size
        self.memory = ReplayMemory(replay_size)

        # nets & optimizer
        self.policy_net = CDQN().to(self.device)
        self.target_net = CDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

    def process_observation(self):
        """3-channel grid: walls, agent, goal."""
        walls_channel = np.zeros((self.env.grid_size, self.env.grid_size), np.float32)
        agent_channel = np.zeros((self.env.grid_size, self.env.grid_size), np.float32)
        goal_channel = np.zeros((self.env.grid_size, self.env.grid_size), np.float32)

        for (i, j) in self.env.wall_positions:
            walls_channel[i, j] = 1.0
        ai, aj = self.env.current_state['player_position']
        agent_channel[ai, aj] = 1.0
        gi, gj = self.env.goal_room
        goal_channel[gi, gj] = 1.0

        return torch.tensor(
            np.stack([walls_channel, agent_channel, goal_channel], axis=0)[None],
            dtype=torch.float32,
            device=self.device
        )

    def select_action(self, state):
        if random.random() < self.epsilon:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device), True
        with torch.no_grad():
            return self.policy_net(state).max(1).indices.view(1, 1), False

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device,
            dtype=torch.bool
        )

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_mask.sum() > 0:
            with torch.no_grad():
                next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states) \
                    .gather(1, next_actions).squeeze(1)

        expected = reward_batch + self.gamma * next_state_values
        loss = F.mse_loss(state_action_values, expected.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # soft update
        for p_target, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            p_target.data.copy_(p_target.data * (1.0 - self.tau) + p.data * self.tau)

        return loss.item()

    def train(self, num_episodes: int, log_dir: str):
        writer = SummaryWriter(log_dir=log_dir)
        total_loss = 0.0
        step_count = 0

        for ep in range(num_episodes):
            obs, reward, done, _ = self.env.reset()
            state = self.process_observation()
            ep_reward, ep_steps = 0, 0

            writer.add_scalar("Epsilon/episode", self.epsilon, ep)
            for t in count():
                action, _ = self.select_action(state)
                obs, reward, done, _ = self.env.step(action)
                r_tensor = torch.tensor([reward], device=self.device)
                ep_reward += reward

                next_state = None if done else self.process_observation()
                self.memory.push(state, action, next_state, r_tensor)
                state = next_state

                loss = self.optimize_model()
                step_count += 1
                if loss is not None:
                    total_loss += loss
                    writer.add_scalar("Loss/step", total_loss / step_count, step_count)

                ep_steps += 1
                if done:
                    writer.add_scalar("Reward/episode", ep_reward / ep_steps, ep + 1)
                    writer.add_scalar("Step/episode", step_count / (ep + 1), ep + 1)
                    writer.add_scalar("Loss/episode_avg", total_loss / (ep + 1), ep + 1)
                    print(f"Episode {ep + 1}: steps={ep_steps}, reward={ep_reward:.2f}, ε={self.epsilon:.4f}")
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        writer.close()
        fname = f"cnn_dqn_model_{int(time.time())}.pth"
        torch.save(self.policy_net.state_dict(), fname)
        print(f"Training complete, model saved to {fname}")

    def play(self, num_episodes: int, model_path: str):
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()

        for ep in range(num_episodes):
            obs, _, done, _ = self.env.reset()
            refresh(obs, 0, done, {})
            state = self.process_observation()
            total_reward = 0

            while not done:
                with torch.no_grad():
                    action = self.policy_net(state).max(1).indices.view(1, 1)
                obs, reward, done, _ = self.env.step(action)
                refresh(obs, reward, done, {})
                total_reward += reward
                state = self.process_observation()

            print(f"Eval Episode {ep + 1}: Reward = {total_reward}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate a CNN-DQN agent on grid world.")
    parser.add_argument('--mode', choices=['train', 'eval'], default='eval')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--gui', action='store_true', default=True)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_min', type=float, default=0.1)
    parser.add_argument('--epsilon_decay', type=float, default=0.9995)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--replay_size', type=int, default=10000)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="cnn_dqn_model_20250404-234200.pth")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    agent = CDQNAgent(
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
        agent.play(args.episodes, args.model_path)
