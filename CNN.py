"""
Deep Q-Network (DQN) Implementation for Grid-Based Environment with CNN
"""
import random
from collections import namedtuple, deque
from datetime import datetime
from itertools import count

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from vis_gym import *


# ================ CONFIGURATION ================

class Config:
    """Modified configuration parameters for 8x8 grid"""

    # Environment settings
    GUI_ENABLED = True  # Can be False since we're using grid representation now

    # Hardware settings
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # CNN input settings
    IMAGE_HEIGHT = 10  # Native grid size
    IMAGE_WIDTH = 10  # Native grid size
    # Removed stacked frames as we're using a 3-channel representation

    # DQN hyperparameters
    BATCH_SIZE = 64  # Reduced batch size for smaller network
    GAMMA = 0.9
    EPSILON_START = 1
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 0.99
    TAU = 0.005
    LEARNING_RATE = 5e-4  # Slightly increased learning rate

    # Training settings
    REPLAY_MEMORY_SIZE = 10000
    TRAIN_EPISODES = 700
    EVAL_EPISODES = 10

    # Data structures
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# ================ ENVIRONMENT SETUP ================

def setup_environment():
    """Initialize the environment"""
    setup(GUI=Config.GUI_ENABLED)
    env = game
    n_actions = len(env.actions)

    print(f"Using device: {Config.DEVICE}")

    return env, n_actions


# ================ NEURAL NETWORK ================

"""
Modified CNNDQN class with debug prints to find the tensor shape issue
"""

"""
Simple fix for the CNN-DQN class - modify only the linear layer size
"""


class CNNDQN(nn.Module):
    """Modified CNN-Based Deep Q-Network Model for smaller 8x8 grid inputs."""

    def __init__(self, h, w, outputs):
        super(CNNDQN, self).__init__()

        # Input has 3 channels (walls, agent, goal)
        input_channels = 3

        # Smaller kernel sizes and no stride for the smaller input grid
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(8)  # Added batch normalization

        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(16)  # Added batch normalization

        # Calculate the correct input size for the linear layer
        # For 8x8 input with the above architecture:
        # After conv1: 7x7
        # After pool1 (stride=1): 6x6
        # After conv2: 5x5
        # So the flattened size is 32 * 5 * 5 = 800
        linear_input_size = 784

        self.fc1 = nn.Linear(linear_input_size, 256)  # Reduced size of first FC layer
        self.fc2 = nn.Linear(256, outputs)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # First convolutional layer
        x = F.relu(self.bn1(self.conv1(x)))

        # Max pooling with stride=1 to avoid reducing dimensions too quickly
        x = F.max_pool2d(x, kernel_size=2, stride=1)

        # Second convolutional layer
        x = F.relu(self.bn2(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)


# ================ REPLAY MEMORY ================

class ReplayMemory:
    """Experience replay buffer"""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Config.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ================ AGENT ================

class CNNDQNAgent:
    """CNN-DQN Agent implementation"""

    step_counter = 0

    def __init__(self, env, n_actions):
        self.env = env
        self.n_actions = n_actions
        self.epsilon = Config.EPSILON_START

        # Initialize networks
        self.policy_net = CNNDQN(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, n_actions).to(Config.DEVICE)
        self.target_net = CNNDQN(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, n_actions).to(Config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize optimizer and memory
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=Config.LEARNING_RATE, amsgrad=True)
        self.memory = ReplayMemory(Config.REPLAY_MEMORY_SIZE)

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            action = torch.tensor([[self.env.action_space.sample()]],
                                  device=Config.DEVICE, dtype=torch.long)
            is_random = True
        else:
            with torch.no_grad():
                action = self.policy_net(state).max(1).indices.view(1, 1)
            is_random = False
        return action, is_random

    def get_screen(self):
        """Capture screenshot from the Pygame display"""
        surface = pygame.display.get_surface()

        if surface is None:
            return None

        pixels = pygame.surfarray.array3d(surface)

        # Convert to grayscale
        gray_screen = np.mean(pixels, axis=2).astype(np.uint8)

        # Resize to the desired dimensions
        surf = pygame.surfarray.make_surface(gray_screen)
        resized_surf = pygame.transform.scale(surf, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
        resized_screen = pygame.surfarray.array2d(resized_surf).astype(np.float32)

        # Normalize pixel values to [0, 1]
        normalized_screen = resized_screen / 255.0

        # Convert to PyTorch tensor
        screen_tensor = torch.tensor(normalized_screen, dtype=torch.float32, device=Config.DEVICE)
        return screen_tensor.unsqueeze(0)  # Add batch dimension

    def stack_frames(self, new_frame, reset=False):
        """Stack frames to create the state"""
        if reset or len(self.frame_buffer) == 0:
            # Initialize buffer with copies of the first frame
            for _ in range(Config.STACKED_FRAMES):
                self.frame_buffer.append(new_frame)
        else:
            # Add new frame to the buffer
            self.frame_buffer.append(new_frame)

        # Stack frames into a single tensor [stacked_frames, H, W]
        stacked_frames = torch.cat(list(self.frame_buffer), dim=0)
        return stacked_frames.unsqueeze(0)  # Add batch dimension

    def process_observation(self):
        """
        Create a multi-channel grid representation:
          - Channel 0: Walls (1 for wall, 0 otherwise)
          - Channel 1: Agent (1 at the agent's position, 0 otherwise)
          - Channel 2: Goal (1 at the goal position, 0 otherwise)
        """
        grid_size = self.env.grid_size

        # Initialize empty grids for each channel
        walls_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
        agent_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
        goal_channel = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Fill in the walls channel
        for pos in self.env.wall_positions:
            i, j = pos
            walls_channel[i, j] = 1.0

        # Mark the agent's position
        agent_i, agent_j = self.env.current_state['player_position']
        agent_channel[agent_i, agent_j] = 1.0

        # Mark the goal position
        goal_i, goal_j = self.env.goal_room
        goal_channel[goal_i, goal_j] = 1.0

        # Stack the channels: resulting shape will be [3, grid_size, grid_size]
        grid_representation = np.stack([walls_channel, agent_channel, goal_channel], axis=0)
        return grid_representation

    def get_env_tensor(self):
        """Convert the processed observation into a tensor for the CNN.
        This version works directly with the 8x8 grid without resizing.
        """
        grid_matrix = self.process_observation()  # Shape: [3, grid_size, grid_size]

        # Convert to PyTorch tensor
        grid_tensor = torch.tensor(grid_matrix, dtype=torch.float32, device=Config.DEVICE)

        # No need to resize if we're using the native 8x8 grid
        # Just add a batch dimension: [1, 3, 8, 8]
        return grid_tensor.unsqueeze(0)

    def get_state_tensor(self, reset=False):
        """Obtain the current state as a tensor without frame stacking."""
        return self.get_env_tensor()

    def optimize_model(self):
        """Perform one step of optimization with Double DQN implementation"""
        if len(self.memory) < Config.BATCH_SIZE:
            return None

        # Sample transitions
        transitions = self.memory.sample(Config.BATCH_SIZE)
        batch = Config.Transition(*zip(*transitions))

        # Create mask for non-final states
        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            device=Config.DEVICE, dtype=torch.bool
        )

        # Prepare batch data
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Get current Q values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute target Q values using Double DQN approach
        next_state_values = torch.zeros(Config.BATCH_SIZE, device=Config.DEVICE)

        if non_final_mask.sum() > 0:  # Only if there are non-final states
            with torch.no_grad():
                # Double DQN: use policy_net to select actions
                next_action_indices = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                # Use target_net to evaluate those actions
                next_state_values[non_final_mask] = self.target_net(
                    non_final_next_states).gather(1, next_action_indices).squeeze(1)

        # Compute expected Q values
        expected_state_action_values = (next_state_values * Config.GAMMA) + reward_batch

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def soft_update_target_network(self):
        """Soft update target network"""
        target_dict = self.target_net.state_dict()
        policy_dict = self.policy_net.state_dict()

        for key in policy_dict:
            target_dict[key] = policy_dict[key] * Config.TAU + target_dict[key] * (1 - Config.TAU)

        self.target_net.load_state_dict(target_dict)

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(Config.EPSILON_MIN, self.epsilon * Config.EPSILON_DECAY)

    def save_model(self, suffix=""):
        """Save the trained model"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_filename = f'cnn_dqn_model_{timestamp}{suffix}.pth'
        torch.save(self.policy_net.state_dict(), model_filename)
        return model_filename

    def load_model(self, filename):
        """Load a saved model"""
        self.policy_net.load_state_dict(torch.load(filename, map_location=Config.DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ================ TRAINING ================

def train(agent, num_episodes, writer):
    """Train the agent"""
    # Track all losses for global metrics
    total_loss = 0.0
    steps = 1

    for episode in range(num_episodes):
        # Reset environment
        obs, reward, done, info = agent.env.reset()

        # Get initial state
        state = agent.get_state_tensor(reset=True)

        if state is None:
            print("Warning: Could not get initial screen. Ensure GUI is enabled.")
            continue

        # Episode tracking
        episode_reward = 0
        episode_step = 1

        # Log epsilon
        writer.add_scalar("Epsilon/episode", agent.epsilon, episode)

        # Episode loop
        for t in count():
            # Select and perform action
            action, _ = agent.select_action(state)
            obs, reward, done, info = agent.env.step(action)

            # Update display using vis_gym's refresh function
            # refresh(obs, reward, done, info)

            reward_tensor = torch.tensor([reward], device=Config.DEVICE)
            episode_reward += reward

            # Process next state
            next_state = None if done else agent.get_state_tensor()

            # Store transition and optimize
            agent.memory.push(state, action, next_state, reward_tensor)
            state = next_state

            # Optimize model and track loss
            loss_value = agent.optimize_model()
            steps += 1
            if loss_value is not None:
                total_loss += loss_value

                # Log step-wise loss (per optimization step)
                writer.add_scalar("Loss/step", total_loss / steps, steps)

            # Update target network
            agent.soft_update_target_network()
            episode_step += 1

            # Episode end handling
            if done:
                # Calculate and log episode metrics
                writer.add_scalar("Reward/episode", episode_reward / episode_step, episode + 1)
                writer.add_scalar("Step/episode", steps / (episode + 1), episode + 1)
                writer.add_scalar("Loss/episode_avg", total_loss / (episode + 1), episode + 1)

                print(
                    f"Episode {episode + 1}: {episode_step} steps, Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}"
                )
                break

        # Decay epsilon
        agent.decay_epsilon()

    # Save model
    return agent.save_model()


# ================ EVALUATION ================

def evaluate(agent, num_episodes):
    """Evaluate the agent"""
    agent.policy_net.eval()

    for episode in range(num_episodes):
        obs, reward, done, info = agent.env.reset()
        refresh(obs, reward, done, info)
        state = agent.get_state_tensor(reset=True)
        total_reward = 0

        while True:
            # Select best action
            with torch.no_grad():
                action = agent.policy_net(state).max(1).indices.view(1, 1)

            # Step environment
            obs, reward, done, info = agent.env.step(action)

            # Update display
            refresh(obs, reward, done, info)

            total_reward += reward

            if done:
                print(f"Eval Episode {episode + 1}: Reward = {total_reward}")
                break

            state = agent.get_state_tensor()


# ================ MAIN FUNCTION ================

def main():
    """Main entry point"""
    # Setup
    TRAIN_MODE = False

    env, n_actions = setup_environment()
    agent = CNNDQNAgent(env, n_actions)

    if TRAIN_MODE:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f"runs/cnn_dqn_{timestamp}"
        writer = SummaryWriter(log_dir=log_dir)
        model_path = train(agent, Config.TRAIN_EPISODES, writer)
        writer.close()
        print(f"Training completed. Model saved to {model_path}")
    else:
        agent.load_model('cnn_dqn_model_20250404-174434.pth')
        evaluate(agent, Config.EVAL_EPISODES)


if __name__ == "__main__":
    main()
