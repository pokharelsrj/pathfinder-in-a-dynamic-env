import pygame
import numpy as np
import sys
import gymnasium as gym
from gymnasium import spaces
import random
import math

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
        self.size = 10 # 5x5 grid
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
        self.RED = 2   # -1000 penalty
        self.ORANGE = 3  # -500 penalty
        
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
            self.cell_size = 500/self.size  # Size of each cell in pixels
            pygame.init()
            # Make window taller to accommodate reward display and controls below the grid
            self.window_width = self.size * self.cell_size
            self.window_height = self.size * self.cell_size + 80  # Extra space for display
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Dynamic Maze Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)
            self.goal_font = pygame.font.SysFont(None, 48)

        
        
    def _create_grid(self):
        """Initialize the grid with walls and colored cells"""
        # Create an empty grid
        grid = np.zeros((self.size, self.size), dtype=np.int32)
        
        # Number of each obstacle type (adjustable)
        num_walls = int(0.25 * (self.size ** 2)) # 25% walls
        num_red = int(0 * (self.size ** 2))  # 10% red cells
        num_orange = int(0 * (self.size ** 2)) # 10% orange cells
        
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
    
    def draw_brick_pattern(self, surface, rect):
        """
        Draw a brick pattern inside the given rectangle on the specified surface.
        Args:
        surface: The pygame surface to draw on
        rect: The pygame.Rect to fill with the brick pattern
        """
        # Colors
        brick_color = (139, 69, 19)  # Brown for bricks
        mortar_color = (169, 169, 169)  # Gray for mortar
    
        # First draw the base background (mortar color)
        pygame.draw.rect(surface, mortar_color, rect)
    
        # Brick size and mortar parameters
        mortar_thickness = max(1, min(rect.width, rect.height) // 25)
        brick_height = rect.height // 3
        brick_width = rect.width // 2
    
        # Draw the brick pattern rows
        for row in range((rect.height // brick_height) + 1):
            y_pos = rect.y + row * brick_height
        
            # Offset every other row for the brick pattern
            x_offset = brick_width // 2 if row % 2 else 0
        
            # Draw the bricks in this row
            for col in range(-1, (rect.width // brick_width) + 2):
                x_pos = rect.x + x_offset + col * brick_width
            
                # Define brick rectangle with mortar spacing
                brick = pygame.Rect(
                    x_pos + mortar_thickness,
                    y_pos + mortar_thickness,
                    brick_width - (mortar_thickness * 2),
                    brick_height - (mortar_thickness * 2)
                )
            
                # Only draw the brick if it's at least partially inside the cell
                if (brick.right > rect.left and brick.left < rect.right and
                    brick.bottom > rect.top and brick.top < rect.bottom):
                
                    # Clip the brick to stay within the cell boundaries
                    if brick.left < rect.left:
                        brick.width -= (rect.left - brick.left)
                        brick.left = rect.left
                    if brick.right > rect.right:
                        brick.width = rect.right - brick.left
                    if brick.top < rect.top:
                        brick.height -= (rect.top - brick.top)
                        brick.top = rect.top
                    if brick.bottom > rect.bottom:
                        brick.height = rect.bottom - brick.top
                
                    # Draw the brick only if it has positive dimensions
                    if brick.width > 0 and brick.height > 0:
                        pygame.draw.rect(surface, brick_color, brick)

    def draw_agent(self):
        """
        Draw a simple robot character using basic shapes
        """
        # Calculate center position
        center_x = self.agent_pos[1] * self.cell_size + self.cell_size // 2
        center_y = self.agent_pos[0] * self.cell_size + self.cell_size // 2
    
        # Robot size (slightly smaller than the cell)
        size = self.cell_size // 3
    
        # Robot head (square)
        head_color = (100, 100, 180)  # Steel blue
        head_rect = pygame.Rect(
            center_x - size,
            center_y - size,
            size * 2,
            size * 2
        )
        pygame.draw.rect(self.window, head_color, head_rect)
    
        # Robot eyes (two small rectangles)
        eye_color = (255, 255, 0)  # Yellow
        eye_width = size // 2
        eye_height = size // 3
        eye_y = center_y - size // 2
    
        # Left eye
        left_eye_rect = pygame.Rect(
            center_x - size // 2 - eye_width // 2,
            eye_y,
            eye_width,
            eye_height
        )
        pygame.draw.rect(self.window, eye_color, left_eye_rect)
    
        # Right eye
        right_eye_rect = pygame.Rect(
            center_x + size // 2 - eye_width // 2,
            eye_y,
            eye_width,
            eye_height
        )
        pygame.draw.rect(self.window, eye_color, right_eye_rect)
    
        # Antenna
        antenna_color = (200, 0, 0)  # Red
        pygame.draw.line(
            self.window,
            antenna_color,
            (center_x, center_y - size),
            (center_x, center_y - size - size//2),
            3
        )
    
        # Antenna top
        pygame.draw.circle(
            self.window,
            antenna_color,
            (center_x, center_y - size - size//2),
            size // 4
        )
    
        # Robot mouth (simple line)
        mouth_y = center_y + size // 2
        pygame.draw.line(
            self.window,
            (50, 50, 50),
            (center_x - size // 2, mouth_y),
            (center_x + size // 2, mouth_y),
            2
        )
    
        # Optional: Add simple body outline
        body_color = (70, 70, 150)  # Darker blue
        body_top = center_y + size
        body_height = size
    
        # Body rectangle
        body_rect = pygame.Rect(
            center_x - size * 0.75,
            body_top,
            size * 1.5,
            body_height
        )
        pygame.draw.rect(self.window, body_color, body_rect)

    def render(self):
        """Render the environment with an enhanced UI"""
        if self.render_mode is None:
            return
    
        if self.render_mode == 'human':
            # Background gradient
            for y in range(int(self.window_height)):
                color_value = max(200, 255 - (y // 4))
                pygame.draw.line(
                    self.window, 
                    (color_value, color_value, 255), 
                    (0, y), 
                    (self.window_width, y)
                )
        
            # Create a semi-transparent overlay for the grid area
            grid_surface = pygame.Surface((self.size * self.cell_size, self.size * self.cell_size), pygame.SRCALPHA)
            grid_surface.fill((255, 255, 255, 180))  # Semi-transparent white
        
            # Draw the grid cells on the overlay
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
                        self.draw_brick_pattern(grid_surface, rect)
                    elif cell_type == self.RED:
                        pygame.draw.rect(grid_surface, (255, 100, 100, 220), rect)  # Red cells
                    elif cell_type == self.ORANGE:
                        pygame.draw.rect(grid_surface, (255, 165, 0, 220), rect)  # Orange cells
                    else:
                        pygame.draw.rect(grid_surface, (230, 230, 230, 180), rect)  # Light gray for empty
                
                    # Draw grid lines with some depth
                    pygame.draw.rect(grid_surface, (100, 100, 100, 150), rect, 1)
                    # Inner shadow for 3D effect
                    pygame.draw.line(grid_surface, (50, 50, 50, 80), 
                                (rect.left, rect.bottom-1), (rect.right, rect.bottom-1), 1)
                    pygame.draw.line(grid_surface, (50, 50, 50, 80), 
                                (rect.right-1, rect.top), (rect.right-1, rect.bottom), 1)
        
            # Blit the grid surface onto the main window
            self.window.blit(grid_surface, (0, 0))
        
            # Mark start position with 'S' in a fancy circle
            start_circle_radius = self.cell_size // 4
            pygame.draw.circle(
                self.window, 
                (0, 128, 0),  # Green
                (self.start_pos[1] * self.cell_size + self.cell_size // 2,
                self.start_pos[0] * self.cell_size + self.cell_size // 2),
                start_circle_radius
            )
            start_text = self.font.render('S', True, (255, 255, 255))
            start_rect = start_text.get_rect(center=(
            self.start_pos[1] * self.cell_size + self.cell_size // 2,
            self.start_pos[0] * self.cell_size + self.cell_size // 2
            ))
            self.window.blit(start_text, start_rect)
        
            # Mark goal position with 'G' in a fancy star-like shape
            goal_center = (
                self.goal_pos[1] * self.cell_size + self.cell_size // 2,
                self.goal_pos[0] * self.cell_size + self.cell_size // 2
            )
            goal_radius = self.cell_size // 3
        
            # Draw a simple star (or target) for the goal
            pygame.draw.circle(self.window, (180, 0, 0), goal_center, goal_radius, 2)
            pygame.draw.circle(self.window, (180, 0, 0), goal_center, goal_radius // 2)
            pygame.draw.line(self.window, (180, 0, 0), 
                        (goal_center[0] - goal_radius, goal_center[1]),
                        (goal_center[0] + goal_radius, goal_center[1]), 2)
            pygame.draw.line(self.window, (180, 0, 0), 
                        (goal_center[0], goal_center[1] - goal_radius),
                        (goal_center[0], goal_center[1] + goal_radius), 2)
        
            # Draw the agent
            self.draw_agent()
        
            # Create a fancy panel for controls and status
            panel_height = 80
            panel_rect = pygame.Rect(0, self.size * self.cell_size, self.window_width, panel_height)
            panel_surface = pygame.Surface((self.window_width, panel_height), pygame.SRCALPHA)
            panel_surface.fill((30, 30, 60, 220))  # Semi-transparent dark blue
        
            # Add a decorative top border to the panel
            pygame.draw.line(panel_surface, (100, 100, 255), 
                         (0, 0), (self.window_width, 0), 3)
        
            # Display reward with a fancy format centered at the top of the panel
            reward_font = pygame.font.SysFont(None, 28)
            reward_text = reward_font.render(f"SCORE: {self.total_reward}", True, (255, 255, 100))
            reward_rect = reward_text.get_rect(center=(self.window_width // 2, 20))
            panel_surface.blit(reward_text, reward_rect)
        
            # Display controls with a fancier format, all center-aligned
            controls_font = pygame.font.SysFont(None, 22)
            controls_text1 = controls_font.render("CONTROLS:", True, (200, 200, 255))
            controls_text2 = controls_font.render("W, A, S, D to move", True, (255, 255, 255))
            controls_text3 = controls_font.render("R to reset", True, (255, 255, 255))
        
            controls_rect1 = controls_text1.get_rect(center=(self.window_width // 2, 40))
            controls_rect2 = controls_text2.get_rect(center=(self.window_width // 2, 55))
            controls_rect3 = controls_text3.get_rect(center=(self.window_width // 2, 70))
        
            panel_surface.blit(controls_text1, controls_rect1)
            panel_surface.blit(controls_text2, controls_rect2)
            panel_surface.blit(controls_text3, controls_rect3)
        
            # Blit the panel surface onto the main window
            self.window.blit(panel_surface, (0, self.size * self.cell_size))
        
            # Display "GOAL REACHED!" when goal is reached
            if self.goal_reached:
                # Create a pulsating effect
                pulse_value = abs(math.sin(pygame.time.get_ticks() / 300)) * 255
                goal_surface = pygame.Surface((self.window_width, 40), pygame.SRCALPHA)
                goal_surface.fill((0, 100, 0, min(200, int(pulse_value))))
            
                goal_reached_text = self.goal_font.render("GOAL REACHED!", True, (255, 255, 0))
                goal_reached_rect = goal_reached_text.get_rect(center=(self.window_width // 2, 20))
                goal_surface.blit(goal_reached_text, goal_reached_rect)
            
                # Position at the bottom of the grid, above the panel
                self.window.blit(goal_surface, (0, self.size * self.cell_size - 40))
            
                # Add celebration particles
                if random.random() < 0.3:  # Only create particles occasionally
                    for _ in range(5):
                        x = random.randint(0, self.window_width)
                        y = random.randint(self.size * self.cell_size - 60, self.size * self.cell_size - 20)
                        size = random.randint(2, 5)
                        color = random.choice([(255,255,0), (0,255,0), (0,255,255), (255,0,255)])
                        pygame.draw.circle(self.window, color, (x, y), size)
        
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
                    if event.key == pygame.K_w:
                        self._take_step(0)  # up
                    elif event.key == pygame.K_d:
                        self._take_step(1)  # right
                    elif event.key == pygame.K_s:
                        self._take_step(2)  # down
                    elif event.key == pygame.K_a:
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