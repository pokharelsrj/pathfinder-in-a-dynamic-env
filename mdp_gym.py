import gym
from gym import spaces
import numpy as np
import random


class CastleEscapeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CastleEscapeEnv, self).__init__()
        self.grid_size = 5
        self.rooms = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.goal_room = (4, 4)
        self.randomise_counter = 0

        # Define health states
        self.health_states = ['Full', 'Injured', 'Critical']
        self.health_state_to_int = {'Full': 2, 'Injured': 1, 'Critical': 0}
        self.int_to_health_state = {2: 'Full', 1: 'Injured', 0: 'Critical'}

        # Rewards
        self.rewards = {
            'goal': 10000,
            'wall_hit': -1000
        }

        # Only movement actions are available.
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space includes player's position, wall indicator, and health.
        obs_space_dict = {
            'player_position': spaces.Tuple((spaces.Discrete(self.grid_size), spaces.Discrete(self.grid_size))),
            'wall_in_cell': spaces.Discrete(2),
            'player_health': spaces.Discrete(len(self.health_states)),
        }
        self.observation_space = spaces.Dict(obs_space_dict)

        # Walls configuration
        self.num_walls = 3
        self.wall_positions = []

        self.reset()

    def randomise_walls(self):
        # Exclude start and goal positions.
        available_positions = [pos for pos in self.rooms if pos not in [(0, 0), self.goal_room]]
        if len(available_positions) < self.num_walls:
            return available_positions
        else:
            return random.sample(available_positions, self.num_walls)

    def reset(self):
        """Resets the game to the initial state."""
        self.current_state = {
            'player_position': (0, 0),
            'player_health': 'Full'
        }
        self.wall_positions = self.randomise_walls()
        return self.get_observation(), 0, False, {}

    def move_player_to_random_adjacent(self):
        """Move player to a random adjacent cell without going out of bounds"""
        x, y = self.current_state['player_position']
        directions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        # Filter out-of-bounds positions
        adjacent_positions = [
            pos for pos in directions
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
        ]

        # Move player to a random adjacent position
        if adjacent_positions:
            self.current_state['player_position'] = random.choice(adjacent_positions)

    def get_observation(self):
        wall_in_cell = 1 if self.current_state['player_position'] in self.wall_positions else 0
        obs = {
            'player_position': self.current_state['player_position'],
            'wall_in_cell': wall_in_cell,
            'player_health': self.health_state_to_int[self.current_state['player_health']]
        }
        return obs

    def is_terminal(self):
        # Reaching the goal is victory.
        if self.current_state['player_position'] == self.goal_room:  # Reaching the goal means victory
            return 'goal'
        if self.current_state['player_health'] == 'Critical':  # Losing health 3 times results in defeat
            return 'defeat'
        return False

    def move_player(self, action):
        """Moves the player according to the action provided."""
        x, y = self.current_state['player_position']
        directions = {
            'UP': (x - 1, y),
            'DOWN': (x + 1, y),
            'LEFT': (x, y - 1),
            'RIGHT': (x, y + 1)
        }
        new_position = directions.get(action, (x, y))

        # Ensure new position is within bounds.
        if 0 <= new_position[0] < self.grid_size and 0 <= new_position[1] < self.grid_size:
            # 90% chance to move as intended.
            if random.random() <= 0.9:
                self.current_state['player_position'] = new_position
            else:
                # 10% chance to move to a random adjacent cell.
                adjacent_positions = [pos for pos in directions.values()
                                      if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size]
                if adjacent_positions:
                    self.current_state['player_position'] = random.choice(adjacent_positions)

            # If new position is a wall, decrement health.
            if self.current_state['player_position'] in self.wall_positions:
                if self.current_state['player_health'] == 'Full':
                    self.current_state['player_health'] = 'Injured'
                elif self.current_state['player_health'] == 'Injured':
                    self.current_state['player_health'] = 'Critical'
                self.move_player_to_random_adjacent()
                message = f"Hit a wall at {self.current_state['player_position']}! Health now {self.current_state['player_health']}."
                return message, self.rewards['wall_hit']
            return f"Moved to {self.current_state['player_position']}", 0
        else:
            return "Out of bounds!", 0

    def step(self, action):
        """Performs one step in the environment."""
        # Convert string action to index if needed.
        if isinstance(action, str):
            action = self.actions.index(action)

        self.randomise_counter += 1
        # Update wall positions every 3 moves.
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
        elif terminal_state == 'defeat':
            done = True
            reward += self.rewards['wall_hit']
            result += " You've been defeated!"

        observation = self.get_observation()
        info = {'result': result, 'action': action_name}
        return observation, reward, done, info

    def render(self, mode='human'):
        """Renders the current state."""
        print(
            f"Player position: {self.current_state['player_position']}, Health: {self.current_state['player_health']}")
        print(f"Walls at: {self.wall_positions}")

    def close(self):
        pass
