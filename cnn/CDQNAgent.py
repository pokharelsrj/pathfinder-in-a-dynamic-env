import argparse
import os
import random
import time
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
from cnn.CDQN import CDQN
from cnn.ReplayMemory import ReplayMemory
from env.gui import DynamicMazeGUI
from torch import optim
from torch.utils.tensorboard import SummaryWriter


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
        # Initialize the GUI for the dynamic maze environment
        # gui_flag determines whether to display the environment visually
        self.gui = DynamicMazeGUI(gui_flag)
        self.gui.setup()
        self.env = self.gui.game

        # Set computation device (CPU/GPU) and exploration-exploitation parameters
        # gamma: discount factor for future rewards
        # epsilon: exploration rate (probability of random action)
        # epsilon_min: minimum exploration rate
        # epsilon_decay: rate at which exploration decreases
        # tau: controls the rate of target network update in soft updates
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau

        # Set up training batch parameters and experience replay memory
        # batch_size: number of transitions to sample for each training step
        # memory: replay buffer to store and sample past experiences
        self.batch_size = batch_size
        self.memory = ReplayMemory(replay_size)

        # Initialize the policy and target networks
        # policy_net: used to select actions
        # target_net: used to evaluate actions (provides stability in training)
        # Initially, target_net is a copy of policy_net
        self.policy_net = CDQN().to(self.device)
        self.target_net = CDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Set up the optimizer for training the policy network
        # AdamW is a variant of Adam that performs better weight decay regularization
        # amsgrad=True enables a variant of AdamW with improved convergence properties
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

    def process_observation(self):
        """
        Converts the environment state into a 3-channel representation for the CNN.
        Returns a tensor with dimensions [1, 3, grid_size, grid_size] where:
        - Channel 0: Wall positions (1 where walls exist, 0 elsewhere)
        - Channel 1: Agent position (1 at agent's location, 0 elsewhere)
        - Channel 2: Goal position (1 at goal's location, 0 elsewhere)
        """
        walls_channel = np.zeros((self.env.grid_size, self.env.grid_size), np.float32)
        agent_channel = np.zeros((self.env.grid_size, self.env.grid_size), np.float32)
        goal_channel = np.zeros((self.env.grid_size, self.env.grid_size), np.float32)

        # Fill the channels based on the environment state
        for (i, j) in self.env.wall_positions:
            walls_channel[i, j] = 1.0
        ai, aj = self.env.current_state['player_position']
        agent_channel[ai, aj] = 1.0
        gi, gj = self.env.goal_room
        goal_channel[gi, gj] = 1.0

        # Stack channels and convert to PyTorch tensor
        # [None] adds a batch dimension as the first dimension
        return torch.tensor(
            np.stack([walls_channel, agent_channel, goal_channel], axis=0)[None],
            dtype=torch.float32,
            device=self.device
        )

    def select_action(self, state):
        """
        Selects an action using epsilon-greedy policy.
        Returns:
        - action: the selected action
        - is_random: boolean indicating if this was a random (exploration) action
        
        With probability epsilon, chooses a random action (exploration).
        Otherwise, selects the action with highest Q-value from policy_net (exploitation).
        """
        if random.random() < self.epsilon:
            # Exploration: choose a random action
            return torch.tensor([[self.env.action_space.sample()]], device=self.device), True
        
        # Exploitation: choose the best action according to the policy
        with torch.no_grad():  # Disable gradient computation for inference
            return self.policy_net(state).max(1).indices.view(1, 1), False

    def optimize_model(self):
        """
        Performs one step of optimization on the policy network.
        Implements the Double DQN algorithm to reduce overestimation bias.
        
        Returns the loss value (or None if not enough samples in memory).
        """
        # Skip optimization if we don't have enough samples yet
        if len(self.memory) < self.batch_size:
            return None

        # Sample a batch of transitions from memory
        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.Transition(*zip(*transitions))

        # Create a mask for non-terminal states
        # For terminal states, the next_state is None
        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device,
            dtype=torch.bool
        )

        # Gather non-terminal next states
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        # Prepare batches for state, action, and reward
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q-values for current state-action pairs
        # gathering values for the actions that were actually taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states using Double DQN approach
        # Initialize with zeros for terminal states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        if non_final_mask.sum() > 0:  # If we have any non-terminal states
            with torch.no_grad():
                # Select actions using the policy network (first part of Double DQN)
                next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                
                # Evaluate those actions using the target network (second part of Double DQN)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states) \
                    .gather(1, next_actions).squeeze(1)

        # Compute the expected Q-values: r + γ * V(s_{t+1})
        expected = reward_batch + self.gamma * next_state_values
        
        # Compute the loss between current predictions and expected values
        loss = F.mse_loss(state_action_values, expected.unsqueeze(1))

        # Perform optimization
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        
        # Clip gradients to prevent exploding gradient problem
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()  # Update weights

        # Perform soft update of target network
        # θ′ ← τθ + (1 − τ)θ′
        # where θ′ are target network parameters and θ are policy network parameters
        for p_target, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            p_target.data.copy_(p_target.data * (1.0 - self.tau) + p.data * self.tau)

        return loss.item()  # Return the loss value for logging

    def train(self, num_episodes: int, log_dir: str):
        """
        Trains the agent for a specified number of episodes.
        Tracks metrics like rewards, steps, and loss using TensorBoard.
        Saves the final model when training is complete.
        
        Args:
            num_episodes: Number of episodes to train for
            log_dir: Directory for TensorBoard logs
        """
        # Set up TensorBoard logging
        writer = SummaryWriter(log_dir=log_dir)
        total_loss = 0.0
        step_count = 0

        # Training loop for each episode
        for ep in range(num_episodes):
            # Reset environment and get initial state
            obs, reward, done, _ = self.env.reset()
            state = self.process_observation()
            ep_reward, ep_steps = 0, 0

            # Log the current exploration rate
            writer.add_scalar("Epsilon/episode", self.epsilon, ep)
            
            # Training loop for each step in the episode
            for t in count():  # count() is an infinite iterator
                # Select and perform an action
                action, _ = self.select_action(state)
                obs, reward, done, _ = self.env.step(action)
                r_tensor = torch.tensor([reward], device=self.device)
                ep_reward += reward

                # Store the transition in memory and prepare next state
                next_state = None if done else self.process_observation()
                self.memory.push(state, action, next_state, r_tensor)
                state = next_state

                # Optimize the policy network
                loss = self.optimize_model()
                step_count += 1
                
                # Log the loss if optimization happened
                if loss is not None:
                    total_loss += loss
                    writer.add_scalar("Loss/step", total_loss / step_count, step_count)

                # Update episode statistics
                ep_steps += 1
                
                # End the episode when done (goal reached or max steps)
                if done:
                    # Log episode metrics
                    writer.add_scalar("Reward/episode", ep_reward / ep_steps, ep + 1)
                    writer.add_scalar("Step/episode", step_count / (ep + 1), ep + 1)
                    writer.add_scalar("Loss/episode_avg", total_loss / (ep + 1), ep + 1)
                    print(f"Episode {ep + 1}: steps={ep_steps}, reward={ep_reward:.2f}, ε={self.epsilon:.4f}")
                    break

            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Clean up and save model
        writer.close()
        fname = f"cnn_dqn_model_{int(time.time())}.pth"
        torch.save(self.policy_net.state_dict(), fname)
        print(f"Training complete, model saved to {fname}")

    def play(self, num_episodes: int, model_path: str):
        """
        Loads a trained model and runs it in the environment for evaluation.
        No exploration or training happens here - just executing the learned policy.
        
        Args:
            num_episodes: Number of episodes to play
            model_path: Path to the saved model file
        """
        # Load the trained model
        path = os.path.join(os.path.dirname(__file__), "trained_model", model_path)
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()  # Set network to evaluation mode

        # Play loop for each episode
        for ep in range(num_episodes):
            # Reset environment
            obs, _, done, _ = self.env.reset()
            self.gui.refresh(obs, 0, done, {})
            state = self.process_observation()
            total_reward = 0

            # Loop until episode termination
            while not done:
                # Select action based on policy (no exploration)
                with torch.no_grad():
                    action = self.policy_net(state).max(1).indices.view(1, 1)
                
                # Apply action and observe result
                obs, reward, done, _ = self.env.step(action)
                self.gui.refresh(obs, reward, done, {})
                total_reward += reward
                state = self.process_observation()

            # Report episode results
            print(f"Eval Episode {ep + 1}: Reward = {total_reward}")


if __name__ == '__main__':
    # Set up command line argument parsing for flexible usage
    parser = argparse.ArgumentParser(description="Train or evaluate a CNN-DQN agent on grid world.")
    parser.add_argument('--mode', choices=['train', 'play'], default='play')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--gui', action='store_true')
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

    # Determine the appropriate computational device
    # Tries CUDA (GPU) first, then Apple's Metal (MPS), then falls back to CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # Initialize the agent with parsed command-line parameters
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

    # Run the agent in the specified mode (train or play)
    if args.mode == 'train':
        # For training, create a unique log directory if one isn't specified
        log_dir = args.log_dir or f"runs/cnn_dqn_{int(time.time())}"
        agent.train(args.episodes, log_dir)
    else:
        # For play mode, load and run the specified trained model
        agent.play(args.episodes, args.model_path)