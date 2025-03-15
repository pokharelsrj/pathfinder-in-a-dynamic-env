import gym
from gym import spaces
import numpy as np
import random


class CastleEscapeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CastleEscapeEnv, self).__init__()
        # Define a 5x5 grid (positions from (0,0) to (4,4))
        self.grid_size = 5
        self.rooms = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.goal_room = (4, 4)  # Define the goal room
        self.randomise_counter = 0

        # Rewards
        self.rewards = {
            'goal': 10000,
            'wall_hit': -1000  # Penalty for striking a wall
        }

        # Actions: Only movement actions remain
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space: only player's position and indicator for wall presence
        obs_space_dict = {
            'player_position': spaces.Tuple((spaces.Discrete(self.grid_size), spaces.Discrete(self.grid_size))),
            'wall_in_cell': spaces.Discrete(2)  # 0 for False, 1 for True
        }
        self.observation_space = spaces.Dict(obs_space_dict)

        # Walls configuration
        self.num_walls = 3  # number of walls
        self.wall_positions = []  # will hold wall positions

        # Set initial state
        self.reset()

    def randomise_walls(self):
        # Exclude start and goal positions from being walls
        available_positions = [pos for pos in self.rooms if pos not in [(0, 0), self.goal_room]]
        # Randomly sample positions for walls
        if len(available_positions) < self.num_walls:
            return available_positions
        else:
            return random.sample(available_positions, self.num_walls)

    def reset(self):
        """Resets the game to the initial state"""
        self.current_state = {
            'player_position': (0, 0)
        }
        # Initialise walls; ensure walls are not at the start or goal.
        self.wall_positions = self.randomise_walls()
        return self.get_observation(), 0, False, {}

    def get_observation(self):
        wall_in_cell = 1 if self.current_state['player_position'] in self.wall_positions else 0

        obs = {
            'player_position': self.current_state['player_position'],
            'wall_in_cell': wall_in_cell,
        }
        return obs

    def is_terminal(self):
        """Check if the game has reached a terminal state (goal reached)"""
        if self.current_state['player_position'] == self.goal_room:
            return 'goal'
        return False

    def move_player(self, action):
        """Move the player according to the given action"""
        x, y = self.current_state['player_position']
        directions = {
            'UP': (x - 1, y),
            'DOWN': (x + 1, y),
            'LEFT': (x, y - 1),
            'RIGHT': (x, y + 1)
        }
        new_position = directions.get(action, (x, y))

        # Ensure the new position is within bounds
        if 0 <= new_position[0] < self.grid_size and 0 <= new_position[1] < self.grid_size:
            # 90% chance to move as intended
            if random.random() <= 0.9:
                self.current_state['player_position'] = new_position
            else:
                # 10% chance to move to a random adjacent cell
                adjacent_positions = [
                    directions[act] for act in directions
                    if 0 <= directions[act][0] < self.grid_size and 0 <= directions[act][1] < self.grid_size
                ]
                self.current_state['player_position'] = random.choice(adjacent_positions)

            # Check if the new position is a wall
            if self.current_state['player_position'] in self.wall_positions:
                return f"Hit a wall at {self.current_state['player_position']}!", self.rewards['wall_hit']
            return f"Moved to {self.current_state['player_position']}", 0
        else:
            return "Out of bounds!", 0

    def step(self, action):
        """Performs one step in the environment"""
        # Convert string action to index if needed.
        if isinstance(action, str):
            action = self.actions.index(action)

        self.randomise_counter += 1
        print("Turn:", self.randomise_counter)

        # Update wall positions every 3 moves
        if self.randomise_counter % 3 == 0:
            self.wall_positions = self.randomise_walls()

        action_name = self.actions[action]
        result, reward = self.move_player(action_name)

        done = False
        terminal_state = self.is_terminal()
        if terminal_state == 'goal':
            done = True
            reward += self.rewards['goal']
            result += f" You've reached the goal! {self.rewards['goal']} points!"

        observation = self.get_observation()
        info = {'result': result, 'action': action_name}
        return observation, reward, done, info

    def render(self, mode='human'):
        """Renders the current state"""
        print(f"Player position: {self.current_state['player_position']}")
        print(f"Walls at: {self.wall_positions}")

    def close(self):
        """Performs cleanup"""
        pass
