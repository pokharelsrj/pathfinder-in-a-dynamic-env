import pygame
import numpy as np
import sys
import gymnasium as gym
from gymnasium import spaces
import random

class DynamicGridWorldEnv(gym.Env):
    """
    Business Logic: Dynamic Grid World
    
    Features:
    - Start position: top-left (0, 0)
    - Goal position: top-right (4, 4)
    - Obstacles: walls (impassable) and colored cells (penalties)
    - Red cells: -10 reward
    - Orange cells: -5 reward
    - Obstacles change every 2 agent moves
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode=None):
        self.size = 5  # 5x5 grid
        self.start_pos = (0, 0)  # top-left
        self.goal_pos = (self.size-1, self.size-1)   # top-right
        
        # Action space: 0: up, 1: right, 2: down, 3: left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: current position (row, col)
        self.observation_space = spaces.Box(
            low=0, 
            high=self.size-1, 
            shape=(2,), 
            dtype=np.int32
        )
        
        # Cell types
        self.EMPTY = 0
        self.WALL = 1
        self.RED = 2   # -10 penalty
        self.ORANGE = 3  # -5 penalty
        
        # Rewards
        self.rewards = {
            self.EMPTY: 0,
            self.RED: -1000,
            self.ORANGE: -500,
        }
        
        # Initialize the grid, agent position, and move counter
        self.render_mode = render_mode
        self.move_counter = 0
        self.obstacle_refresh_interval = 2
        self.goal_reached = False
        
        # Initialize pygame if using 'human' render mode
        if self.render_mode == "human":
            self.cell_size = 80  # Size of each cell in pixels
            pygame.init()
            # Make window taller to accommodate reward display and controls below the grid
            self.window_width = self.size * self.cell_size
            self.window_height = self.size * self.cell_size + 80  # Extra space for display
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Dynamic GridWorld")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)
            self.goal_font = pygame.font.SysFont(None, 48)
        
    def _create_grid(self):
        """Initialize the grid with walls and colored cells"""
        # Create an empty grid
        grid = np.zeros((self.size, self.size), dtype=np.int32)
        
        # Number of each obstacle type (adjustable)
        num_walls = random.randint(3, 6)  
        num_red = random.randint(2, 4)
        num_orange = random.randint(2, 4)
        
        # Place walls
        self._place_obstacles(grid, self.WALL, num_walls)
        
        # Place red cells
        self._place_obstacles(grid, self.RED, num_red)
        
        # Place orange cells
        self._place_obstacles(grid, self.ORANGE, num_orange)
        
        # Ensure start and goal positions are empty
        grid[self.start_pos] = self.EMPTY
        grid[self.goal_pos] = self.EMPTY
        
        # Ensure there's a valid path from start to goal
        if not self._has_valid_path(grid):
            # If no valid path, recursively create a new grid
            return self._create_grid()
        
        return grid
    
    def _place_obstacles(self, grid, obstacle_type, count):
        """Place obstacles randomly on the grid"""
        placed = 0
        while placed < count:
            row = random.randint(0, self.size-1)
            col = random.randint(0, self.size-1)
            
            # Skip start and goal positions
            if (row, col) == self.start_pos or (row, col) == self.goal_pos:
                continue
                
            # Place obstacle if position is empty
            if grid[row, col] == self.EMPTY:
                grid[row, col] = obstacle_type
                placed += 1
    
    def _has_valid_path(self, grid):
        """Check if there's a valid path from start to goal using BFS"""
        queue = [self.start_pos]
        visited = set([self.start_pos])
        
        while queue:
            row, col = queue.pop(0)
            
            if (row, col) == self.goal_pos:
                return True
            
            # Check all four directions
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                
                # Check if the position is valid
                if (0 <= new_row < self.size and 
                    0 <= new_col < self.size and 
                    grid[new_row, new_col] != self.WALL and
                    (new_row, new_col) not in visited):
                    queue.append((new_row, new_col))
                    visited.add((new_row, new_col))
        
        return False
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Create a new grid
        self.grid = self._create_grid()
        
        # Reset agent position
        self.agent_pos = self.start_pos
        
        # Reset move counter
        self.move_counter = 0
        
        # Reset total reward
        self.total_reward = 0
        
        # Reset goal reached flag
        self.goal_reached = False
        
        observation = np.array(self.agent_pos, dtype=np.int32)
        info = {}
        
        if self.render_mode == 'human':
            self.render()
            
        return observation, info
    
    def step(self, action):
        """Take a step based on the action"""
        # If goal is already reached, don't allow further movement
        if self.goal_reached:
            observation = np.array(self.agent_pos, dtype=np.int32)
            return observation, 0, True, False, {'cell_type': self.grid[self.agent_pos]}
        
        # Update move counter
        self.move_counter += 1
        
        # Change obstacles every 5 moves
        if self.move_counter % self.obstacle_refresh_interval == 0:
            # Store current position
            current_pos = self.agent_pos
            # Recreate grid with new obstacles
            self.grid = self._create_grid()
            # Ensure agent's current position is walkable
            self.grid[current_pos] = self.EMPTY
        
        # Get current position
        row, col = self.agent_pos
        
        # Calculate new position based on action
        if action == 0:  # up
            new_pos = (max(0, row-1), col)
        elif action == 1:  # right
            new_pos = (row, min(self.size-1, col+1))
        elif action == 2:  # down
            new_pos = (min(self.size-1, row+1), col)
        elif action == 3:  # left
            new_pos = (row, max(0, col-1))
        
        # Check if the new position is a wall
        if self.grid[new_pos] == self.WALL:
            new_pos = self.agent_pos  # Invalid move, stay in place
            reward = -1  # Small penalty for trying to move into a wall
        else:
            # Calculate reward based on cell type
            reward = self.rewards[self.grid[new_pos]]
        
        # Update agent position
        self.agent_pos = new_pos
        
        # Update total reward
        self.total_reward += reward
        
        # Check if the goal is reached
        done = self.agent_pos == self.goal_pos
        
        # Additional reward for reaching the goal (10000 as requested)
        if done:
            reward += 10000
            self.total_reward += 10000
            self.goal_reached = True
        
        observation = np.array(self.agent_pos, dtype=np.int32)
        info = {'cell_type': self.grid[self.agent_pos]}
        
        if self.render_mode == 'human':
            self.render()
            
        return observation, reward, done, False, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        if self.render_mode == 'human':
            self.window.fill((255, 255, 255))  # White background
            
            # Draw the grid cells
            for row in range(self.size):
                for col in range(self.size):
                    cell_type = self.grid[row, col]
                    rect = pygame.Rect(
                        col * self.cell_size, 
                        row * self.cell_size, 
                        self.cell_size, 
                        self.cell_size
                    )
                    
                    # Draw cell based on type
                    if cell_type == self.WALL:
                        pygame.draw.rect(self.window, (50, 50, 50), rect)  # Dark gray for walls
                    elif cell_type == self.RED:
                        pygame.draw.rect(self.window, (255, 100, 100), rect)  # Red cells
                    elif cell_type == self.ORANGE:
                        pygame.draw.rect(self.window, (255, 165, 0), rect)  # Orange cells
                    else:
                        pygame.draw.rect(self.window, (200, 200, 200), rect)  # Light gray for empty
                    
                    # Draw grid lines
                    pygame.draw.rect(self.window, (0, 0, 0), rect, 1)
            
            # Mark start position with 'S'
            start_text = self.font.render('S', True, (0, 0, 0))
            start_rect = start_text.get_rect(center=(
                self.start_pos[1] * self.cell_size + self.cell_size // 2,
                self.start_pos[0] * self.cell_size + self.cell_size // 2
            ))
            self.window.blit(start_text, start_rect)
            
            # Mark goal position with 'G'
            goal_text = self.font.render('G', True, (0, 0, 0))
            goal_rect = goal_text.get_rect(center=(
                self.goal_pos[1] * self.cell_size + self.cell_size // 2,
                self.goal_pos[0] * self.cell_size + self.cell_size // 2
            ))
            self.window.blit(goal_text, goal_rect)
            
            # Draw the agent
            agent_center = (
                self.agent_pos[1] * self.cell_size + self.cell_size // 2,
                self.agent_pos[0] * self.cell_size + self.cell_size // 2
            )
            pygame.draw.circle(self.window, (0, 0, 255), agent_center, self.cell_size // 3)
            
            # Display reward below the grid
            reward_text = self.font.render(f"Reward: {self.total_reward}", True, (0, 0, 0))
            reward_rect = reward_text.get_rect(center=(self.window_width // 2, self.size * self.cell_size + 25))
            self.window.blit(reward_text, reward_rect)
            
            # Display controls below the grid
            controls_text = self.font.render("Controls: Arrow keys to move, R to reset", True, (0, 0, 0))
            controls_rect = controls_text.get_rect(center=(self.window_width // 2, self.size * self.cell_size + 55))
            self.window.blit(controls_text, controls_rect)
            
            # Display "GOAL" when goal is reached
            if self.goal_reached:
                goal_reached_text = self.goal_font.render("GOAL!", True, (0, 128, 0))
                goal_reached_rect = goal_reached_text.get_rect(center=(self.window_width // 2, self.window_height // 2 - 50))
                self.window.blit(goal_reached_text, goal_reached_rect)
            
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
            
        elif self.render_mode == 'rgb_array':
            # This would return an RGB array for visualization libraries
            return np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
            
    def close(self):
        """Close the environment"""
        if self.render_mode == "human":
            pygame.quit()


class PygameGridWorldRunner:
    def __init__(self):
        self.env = DynamicGridWorldEnv(render_mode='human')
        self.obs, _ = self.env.reset()
        self.running = True
    
    def run(self):
        """Main game loop"""
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                # Handle key presses for manual control
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self._take_step(0)  # up
                    elif event.key == pygame.K_RIGHT:
                        self._take_step(1)  # right
                    elif event.key == pygame.K_DOWN:
                        self._take_step(2)  # down
                    elif event.key == pygame.K_LEFT:
                        self._take_step(3)  # left
                    elif event.key == pygame.K_r:
                        self.obs, _ = self.env.reset()
        
        self.env.close()
    
    def _take_step(self, action):
        """Take a step in the environment"""
        obs, reward, done, _, info = self.env.step(action)
        self.obs = obs
        
        # No auto-reset when goal is reached


if __name__ == "__main__":
    # Run the game
    runner = PygameGridWorldRunner()
    runner.run()