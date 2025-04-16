# 🧠🕹️ MazeMaster: A Dynamic Grid World + Deep Reinforcement Learning Playground

**Survival of the Smartest.**  
Welcome to **MazeMaster** – a dynamic, obstacle-ridden grid world where pathfinding meets chaos. This isn’t your average game—it's an evolving, AI-driven experiment testing the wit of agents trained through Q-learning and CNN . Whether you’re here to build, play, or train the next grid-exploring super-agent—**you’ve found the right maze.**

---

## 🚀 Features at a Glance

🧱 **Dynamic Obstacles**  
Walls rearrange every few moves. Strategy today is dead tomorrow.

💉 **Health Mechanics**  
Bump into walls? It’ll cost you. Watch your agent go from Full → Injured → Critical.

🎯 **Smart Goal Placement**  
Randomized targets ensure no two episodes feel the same.

🧠 **Reinforcement Learning Agents**  
- 💾 Classic **Q-learning** with partial observability
- 📦 CNN-powered agent with PyTorch
- 🔍 Local state hashing and multi-channel observations

👀 **Visualization FTW**  
Pygame UI with animated bots, reward displays, particle effects & retro vibes.

---

## 🧪 Core Modules

| Module | Purpose |
|--------|---------|
| `mdp_gym.py` | Custom Gym environment with health, wall logic, rewards |
| `vis_gym.py` | Pygame-powered environment visualization |
| `MFMC.py` | Q-learning agent with partial observation support |
| `CNN.py` | DQN agent with CNN architecture and experience replay |
| `dynenv.py` | Variant environment with red/orange penalty zones |
| `environment.py` | Early prototype with hard/soft wall logic |

---

## 🧠 Agent Intelligence

### ✅ Q-Learning (MFMC)
- Local 3x3 grid hashed into stable state representation
- Epsilon-decay exploration
- Wall-aware decision making

### 🧠 DQN (CNN.py)
- 3-channel grid: walls, player, goal
- CNN model trained via replay buffer & target network
- Full TensorBoard logging and model checkpoints

---

## 🕹️ How to Play


# For manual play (WASD controls)
python vis_gym.py



# Run Q-learning
python MFMC.py


# Train the DQN agent
python CNN.py

---

## 💡 Nerdy Nuggets

- Grid size is customizable (currently 8x8)
- Wall density ~30%
- Partial observability via 3x3/5x5 windows
- DQN supports both CPU, MPS, and CUDA backends
- Particle celebration on goal reach 😎

---

## 📂 Project Tree


.
├── mdp_gym.py       # Core Gym environment
├── vis_gym.py       # Game visualization
├── MFMC.py          # Q-learning loop
├── CNN.py           # Deep Q-Learning with PyTorch
├── dynenv.py        # Alternate env with refreshable obstacles
├── environment.py   # Legacy version with hard/soft wall rules
├── Q_table.pickle   # Trained Q-table (generated)
└── .gitignore




## 📈 Result Demos

| 🎮 Episode | Reward Curve | Action Map |
|-----------|---------------|------------|
| ![goal](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdDkwMHN3b2dmc3ZxbjhzYTZ5emY4amtlczR6eWZyaHczbTgyNnM1dCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Ge86Xwnb0FGXmT4D6o/giphy.gif) | *(TensorBoard logs available)* | *(Coming Soon)* |

---

## 📚 Credits & Nerds Behind the Code

Crafted by a team obsessed with intelligent agents, reward hacks, and retro-style game design.  
Drop us a 🌟 if you love robots learning how to escape unpredictable mazes.

