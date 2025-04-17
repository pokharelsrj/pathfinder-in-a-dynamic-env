import gym
from gym import spaces
import random


class DynamicMazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, random_goal=True, grid_size=8, wall_percent=0.3):
        super(DynamicMazeEnv, self).__init__()
        print(random_goal)
        # Flags and parameters
        self.random_goal = random_goal
        self.grid_size = grid_size
        self.wall_percent = wall_percent

        self.start_pos = None
        self.goal_room = None
        self.rooms = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.randomise_counter = 0

        # Define health states
        self.health_states = ['Full', 'Injured', 'Critical']
        self.health_state_to_int = {'Full': 2, 'Injured': 1, 'Critical': 0}
        self.int_to_health_state = {2: 'Full', 1: 'Injured', 0: 'Critical'}

        # Rewards
        self.rewards = {
            'goal': 10000,
            'wall_hit': -1000,
            'step': 0
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
        self.num_walls = int(self.wall_percent * (self.grid_size ** 2))
        print(self.num_walls)
        self.wall_positions = []

        self.reset()

    def randomise_walls(self):
        # Exclude the start, goal, and the player's current position.
        available_positions = [
            pos for pos in self.rooms
            if pos not in [self.goal_room, self.current_state['player_position']]
        ]
        if len(available_positions) < self.num_walls:
            return available_positions
        else:
            return random.sample(available_positions, self.num_walls)

    def reset(self):
        """Resets the game to the initial state with start and goal positions."""
        # Randomize starting position
        start_pos = (0, 0)

        # Choose goal position based on flag
        if self.random_goal:
            available_goals = [pos for pos in self.rooms if pos != start_pos]
            goal_pos = random.choice(available_goals)
        else:
            goal_pos = (self.grid_size - 1, self.grid_size - 1)

        self.current_state = {
            'player_position': start_pos,
            'player_health': 'Full'
        }
        self.goal_room = goal_pos
        self.start_pos = start_pos

        self.wall_positions = self.randomise_walls()
        return self.get_observation(), 0, False, {}

    def move_player_to_random_adjacent(self):
        """Move player to a random adjacent cell without going out of bounds or into a wall."""
        x, y = self.current_state['player_position']
        potential_moves = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        # Filter out-of-bounds positions and positions that contain a wall.
        valid_moves = [
            pos for pos in potential_moves
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size and pos not in self.wall_positions
        ]

        # Move player to a random adjacent valid position, if available.
        if valid_moves:
            self.current_state['player_position'] = random.choice(valid_moves)
        # If no valid moves exist, the player stays in the same position.

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
        if self.current_state['player_position'] == self.goal_room:
            return 'goal'
        if self.current_state['player_health'] == 'Critical':
            return 'defeat'
        return False

    def move_player(self, action):
        x, y = self.current_state['player_position']
        directions = {
            'UP': (x - 1, y),
            'DOWN': (x + 1, y),
            'LEFT': (x, y - 1),
            'RIGHT': (x, y + 1)
        }
        new_position = directions.get(action, (x, y))

        # Check for out-of-bound move and treat it as hitting an outer wall.
        if not (0 <= new_position[0] < self.grid_size and 0 <= new_position[1] < self.grid_size):
            self.move_player_to_random_adjacent()
            message = f"Attempted to move out-of-bounds at {new_position}, treated as wall hit. Health now {self.current_state['player_health']}."
            return message, self.rewards['wall_hit']

        # 90% chance to move as intended; otherwise, move randomly among valid adjacent positions.
        if random.random() <= 0.9:
            self.current_state['player_position'] = new_position
        else:
            adjacent_positions = [
                pos for pos in directions.values()
                if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
            ]
            if adjacent_positions:
                self.current_state['player_position'] = random.choice(adjacent_positions)

        # Check if the new cell contains a wall.
        if self.current_state['player_position'] in self.wall_positions:
            self.move_player_to_random_adjacent()
            message = f"Hit a wall at {self.current_state['player_position']}! Health now {self.current_state['player_health']}."
            return message, self.rewards['wall_hit']

        return f"Moved to {self.current_state['player_position']}", 0

    def step(self, action):
        """Performs one step in the environment."""
        # Convert string action to index if needed.
        if isinstance(action, str):
            action = self.actions.index(action)

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

        self.randomise_counter += 1
        # Update wall positions every 2 moves.
        if self.randomise_counter % 2 == 0:
            self.wall_positions = self.randomise_walls()

        reward += self.rewards['step']
        observation = self.get_observation()
        info = {'result': result, 'action': action_name}
        return observation, reward, done, info

    def render(self, mode='human'):
        """Renders the current state."""
        print(
            f"Player position: {self.current_state['player_position']}, Health: {self.current_state['player_health']}")
        print(f"Goal position: {self.goal_room}")
        print(f"Walls at: {self.wall_positions}")

    def close(self):
        pass
