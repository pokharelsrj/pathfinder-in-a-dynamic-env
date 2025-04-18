# Pathfinder Agent in Dynamic Environment

<table>
<tr>
<td style="width:60%; vertical-align:top;">

This project focuses on developing a pathfinding agent capable of operating in **dynamic environments** such as forest fires, military operation zones, and construction sites. The agent is trained to adapt to real-time environmental changes, including moving obstacles and evolving hazards.

A custom **OpenAI Gym environment** was developed to simulate these scenarios. We began with a basic **Q-learning** approach for initial experimentation and moved to a **Convolutional Deep Q-Network (DQN)** to enable scalability and performance in more complex settings.

</td>
<td style="width:40%; text-align:center;">

<img src="assets/agent_demo.gif" alt="Agent navigating dynamic environment" width="100%"><br/>
<sub><i>**Agent Learning Visualization:** Agent navigating to goal with trained policy</i></sub>

</td>
</tr>
</table>





## Installation

Follow these steps to set up the project locally:

### 1. Clone the repository

```bash
git clone https://github.com/pokharelsrj/pathfinder-in-a-dynamic-env
cd pathfinder-dynamic-env
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Train the model locally to personalize parameters

### 5. (Express) Use the pre-trained model included in the repository if you would like to quickly run the project without waiting for the model to be trained.

```bash
cnn/trained_model/cnn_dqn_model_20250404-234200.pth
```

## Environment Logic
This environment simulates a grid-based navigation problem where an agent must reach a goal while avoiding walls. The environment features dynamic obstacles and stochastic transitions when collisions occur.

### Grid Structure
- The world is represented as a grid of cells
- Each cell can be either empty, contain a wall, the agent, or the goal

### Agent
- Represented by a blue robot character
- Can move in four directions: *Up, Down, Left, and Right*
- Movement keys: *W (up), S (down), A (left), D (right)*

### Goal
- Represented by a red target icon
- Reaching the goal provides a reward of *+10,000*

### Walls
- Represented by brown brick patterns
- Hitting a wall results in a penalty of *-1000*
- After hitting a wall, the agent is moved to a random adjacent cell
- Walls randomize their positions every two moves, creating a dynamic environment

### Action Space
- Four discrete actions: Up, Down, Left, and Right

### Reward Structure
- Goal reached: +10,000
- Wall collision: -1000
- Other moves: No explicit reward/penalty

## Challenge
The main challenge in this environment is to navigate to the goal while avoiding walls that change positions periodically, requiring adaptive pathfinding strategies. Maximizing reward score was the main objective for our agent.

## Implementation Notes
This environment can be used to test various reinforcement learning algorithms, particularly those that can handle:
- Discrete action spaces
- Sparse rewards
- Dynamic obstacles
- Stochastic transitions


## Agent Intelligence

### Q-Learning
- Uses a **3×3 tunnel vision** around the agent, capturing local surroundings.
- The **local view is hashed together with the goal position** to form the state representation.
- Supports **randomized goal placement** in each episode to encourage generalization.
- Implements **epsilon-greedy exploration** with decay.

### Convolutional Deep Q-Network (C-DQN)
- Processes a **3-channel full-grid input**: walls, agent location, and goal.
- Learns via a **Convolutional Neural Network (CNN)**.
- Trained using:
  - **Experience replay** for efficient sample reuse.
  - A **target network** to stabilize Q-value updates.
- Includes **TensorBoard logging** and **checkpointing** for monitoring and reproducibility.


## Usage

### Q-Learning Agent

```
python q-learning/q-learning.py [options]
```

### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Operating mode: either `train` to train the agent or `play` to use a trained Q-table | `play` |
| `--episodes` | Number of episodes for training or playing | `1000000` |
| `--gui` | Enable graphical interface visualization | Disabled |
| `--gamma` | Discount factor for future rewards in Q-learning algorithm | `0.9` |
| `--epsilon` | Initial exploration rate (probability of taking a random action) | `1.0` |
| `--decay_rate` | Rate at which exploration probability decays per episode | `0.999999` |
| `--qtable` | Path to save or load the Q-table file | None |
| `--fixed_goal` | Use a fixed goal position instead of random placement | Disabled |

### Examples

Train a new agent with GUI enabled:
```
python q-learning/q-learning.py --mode train --episodes 500000 --gui --qtable q-learning/trained_model/new_agent.pickle
```

Play with a pre-trained agent:
```
python q-learning/q-learning.py --mode play --qtable q-learning/trained_model/Q_table_random_goal.pickle --gui
```

Train with custom learning parameters:
```
python q-learning/q-learning.py --mode train --gamma 0.95 --epsilon 0.8 --decay_rate 0.9999
```

The Q-learning agent provides a tabular approach to reinforcement learning, while the CNN module implements a deep Q-network approach for more complex state spaces.


### CNN-Based Deep Q-Network Agent


```
python cnn/CDQN.py [options]
```

### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Operating mode: either `train` to train the agent or `play` to use a trained model | `play` |
| `--episodes` | Number of episodes for training or evaluation | `200` |
| `--gui` | Enable graphical interface visualization | Disabled |
| `--gamma` | Discount factor for future rewards | `0.9` |
| `--epsilon_start` | Initial exploration rate | `1.0` |
| `--epsilon_min` | Minimum exploration rate | `0.1` |
| `--epsilon_decay` | Rate at which exploration probability decays | `0.9995` |
| `--tau` | Target network update rate for soft updates | `0.005` |
| `--lr` | Learning rate for neural network optimizer | `5e-4` |
| `--batch_size` | Number of samples per batch for training | `64` |
| `--replay_size` | Size of the experience replay buffer | `10000` |
| `--log_dir` | Directory for storing training logs | None |
| `--model_path` | Path to save or load the model | `cnn_dqn_model_20250404-234200.pth` |

### Examples

Train a new CNN-DQN agent:
```
python cnn/CDQN.py --mode train --episodes 1000 --gui --model_path cnn/trained_model/new_model.pth
```

Evaluate a pre-trained model:
```
python cnn/CDQN.py --mode play --gui --model_path cnn/trained_model/cnn_dqn_model_20250404-234200.pth
```

Train with custom hyperparameters:
```
python cnn/CDQN.py --mode train --gamma 0.95 --epsilon_start 0.9 --epsilon_min 0.05 --epsilon_decay 0.999 --lr 1e-4
```


The CNN-DQN approach offers enhanced capabilities for handling environments with complex visual inputs compared to the tabular Q-learning approach.


## Project Tree

```
.
├── README.md
├── cnn
│   ├── CDQN.py
│   ├── CDQNAgent.py
│   ├── ReplayMemory.py
│   ├── __init__.py
│   └── trained_model
│       └── cnn_dqn_model_20250404-234200.pth
├── env
│   ├── __init__.py
│   ├── environment.py
│   └── gui.py
├── q-learning
│   ├── __init__.py
│   ├── q-learning.py
│   └── trained_model
│       ├── Q_table_fixed_goal.pickle
│       └── Q_table_random_goal.pickle
└── requirements.txt
```

## References

1. https://rajagopalvenkat.com/teaching/lectures/neural_networks/#/16
2. https://arxiv.org/pdf/2403.04807
3. https://stevenschmatz.gitbooks.io/deep-reinforcement-learning/content/deep-q-networks.html





