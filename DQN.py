"""
Deep Q-Network (DQN) Implementation for Grid-Based Environment
"""

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from vis_gym import *


# ================ CONFIGURATION ================

class Config:
    """Core configuration parameters"""

    # Environment settings
    GUI_ENABLED = False

    # Hardware settings
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # DQN hyperparameters
    BATCH_SIZE = 256
    GAMMA = 0.9
    EPSILON_START = 1.0
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 0.9999
    TAU = 0.005
    LEARNING_RATE = 1e-4

    # Training settings
    REPLAY_MEMORY_SIZE = 50000
    TRAIN_EPISODES = 5000
    EVAL_EPISODES = 10

    # Data structures
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# ================ ENVIRONMENT SETUP ================

def setup_environment():
    """Initialize the environment"""
    setup(GUI=Config.GUI_ENABLED)
    env = game
    n_actions = len(env.actions)
    n_observations = 25

    print(f"Using device: {Config.DEVICE}")

    return env, n_actions, n_observations


# ================ NEURAL NETWORK ================

class DQN(nn.Module):
    """Deep Q-Network Model"""

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.layer_4 = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        return self.layer_4(x)


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

class DQNAgent:
    """DQN Agent implementation"""

    def __init__(self, env, n_observations, n_actions):
        self.env = env
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.epsilon = Config.EPSILON_START

        # Initialize networks
        self.policy_net = DQN(n_observations, n_actions).to(Config.DEVICE)
        self.target_net = DQN(n_observations, n_actions).to(Config.DEVICE)
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

    def process_observation(self):
        """Process observation into flattened grid"""
        grid_size = self.env.grid_size
        player_pos = self.env.current_state['player_position']
        flattened_grid = []

        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) == player_pos:
                    flattened_grid.append(3)  # Player
                elif (i, j) in self.env.wall_positions:
                    flattened_grid.append(1)  # Wall
                elif (i, j) == self.env.goal_room:
                    flattened_grid.append(2)  # Goal
                else:
                    flattened_grid.append(0)  # Empty

        return tuple(flattened_grid)

    def get_state_tensor(self, observation=None):
        """Convert observation to tensor state"""
        if observation is None:
            observation = self.process_observation()
        return torch.tensor(observation, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)

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
        model_filename = f'dqn_model_{timestamp}{suffix}.pth'
        torch.save(self.policy_net.state_dict(), model_filename)
        return model_filename

    def load_model(self, filename):
        """Load a saved model"""
        self.policy_net.load_state_dict(torch.load(filename, map_location=Config.DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ================ TRAINING ================

def train(agent, num_episodes, writer):
    """Train the agent"""

    for episode in range(num_episodes):
        # Reset environment
        obs, reward, done, info = agent.env.reset()
        state = agent.get_state_tensor()

        # Episode tracking
        episode_reward = 0
        episode_loss = 0.0
        episode_step = 0

        # Log epsilon
        writer.add_scalar("Epsilon/episode", agent.epsilon, episode)

        # Episode loop
        for t in count():
            # Select and perform action
            action, _ = agent.select_action(state)
            obs, reward, done, info = agent.env.step(action)
            reward_tensor = torch.tensor([reward], device=Config.DEVICE)
            episode_reward += reward

            # Process next state
            next_state = None if done else agent.get_state_tensor()

            # Store transition and optimize
            agent.memory.push(state, action, next_state, reward_tensor)
            state = next_state

            loss_value = agent.optimize_model()
            if loss_value is not None:
                episode_loss += loss_value

            # Update target network
            agent.soft_update_target_network()
            episode_step += 1

            # Episode end handling
            if done:
                # Log statistics
                writer.add_scalar("AverageReward/episode",
                                  (episode_reward / episode_step if episode_step > 0 else 0),
                                  episode)
                writer.add_scalar("ReplayMemory/Size", len(agent.memory), episode)

                print(
                    f"Episode {episode}: {episode_step} steps, Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}")
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
        state = agent.get_state_tensor()
        total_reward = 0

        while True:
            # Select best action
            with torch.no_grad():
                action = agent.policy_net(state).max(1).indices.view(1, 1)

            # Step environment
            obs, reward, done, info = agent.env.step(action)
            total_reward += reward

            # Update UI if enabled
            if Config.GUI_ENABLED:
                refresh(obs, reward, done, info)

            if done:
                print(f"Eval Episode {episode + 1}: Reward = {total_reward}")
                break

            state = agent.get_state_tensor()


# ================ MAIN FUNCTION ================

def main():
    """Main entry point"""
    # Setup
    TRAIN_MODE = True
    env, n_actions, n_observations = setup_environment()
    agent = DQNAgent(env, n_observations, n_actions)

    # Logging
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/dqn_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)

    if TRAIN_MODE:
        model_path = train(agent, Config.TRAIN_EPISODES, writer)
        writer.close()
        print(f"Training completed. Model saved to {model_path}")
    else:
        agent.load_model('dqn_model.pth')
        evaluate(agent, Config.EVAL_EPISODES)


if __name__ == "__main__":
    main()
