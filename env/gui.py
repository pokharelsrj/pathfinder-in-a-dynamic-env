import pygame
import sys
import time
import random
import math
from env.environment import DynamicMazeEnv  # Import the updated DynamicMazeEnv


class DynamicMazeGUI:
    def __init__(self, gui=True, fixed_goal=False):
        # Initialize the maze environment with configurable goal settings
        # Set fixed_goal=True to keep the goal location constant across episodes
        self.game = DynamicMazeEnv(random_goal=not fixed_goal)
        self.GRID_SIZE = self.game.grid_size
        self.CELL_SIZE = 500 // self.GRID_SIZE  # Determine cell pixel size based on grid size
        self.WIDTH = self.GRID_SIZE * self.CELL_SIZE  # Total window width
        self.HEIGHT = self.GRID_SIZE * self.CELL_SIZE  # Total window height

        # Define color constants for rendering
        self.WHITE = (255, 255, 255)  # Background color
        self.GRAY = (200, 200, 200)   # Secondary UI elements
        self.BLACK = (0, 0, 0)        # Grid lines and borders

        # Pygame rendering elements - initialized as None until setup() is called
        self.screen = None           # Main display surface
        self.game_ended = False      # Track if the game has ended (for victory effects)
        self.action_results = [None, None, None, None, None]  # Track last 5 actions for display

        # Animation control parameters
        self.fps = 10                # Target frames per second
        self.sleeptime = 0.1         # Additional delay between frames
        self.clock = None            # Pygame clock for timing control

        # Initialize the GUI immediately if requested
        if gui:
            self.setup()

    def setup(self):
        """
        Initialize Pygame and create the display window.
        Called automatically if gui=True in __init__, or can be called manually.
        """
        pygame.init()
        # Create window with extra space at bottom for status messages
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT + 50))
        pygame.display.set_caption("Dynamic Maze Environment")

    def position_to_grid(self, position):
        """
        Convert grid coordinates (row, col) to pixel coordinates (x, y).
        Important: Pygame uses (x, y) with (0,0) at top-left,
        but our grid uses (row, col) with (0,0) at top-left.
        """
        row, col = position
        return col * self.CELL_SIZE, row * self.CELL_SIZE

    def draw_grid(self):
        """
        Draw the grid lines to visualize the maze cells.
        Creates a clean grid pattern with 1px black lines.
        """
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            for y in range(0, self.HEIGHT, self.CELL_SIZE):
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)
        # Add a shaded area below the maze for potential future info display
        rect = pygame.Rect(0, self.HEIGHT + 100, self.WIDTH, self.HEIGHT)
        pygame.draw.rect(self.screen, self.GRAY, rect)

    def draw_goal_room(self):
        """
        Visualize the goal room with distinctive concentric circles.
        The three-layer design makes the goal highly visible and attractive.
        """
        x, y = self.position_to_grid(self.game.goal_room)
        center_x = x + self.CELL_SIZE // 2
        center_y = y + self.CELL_SIZE // 2

        # Create a target-like pattern with three concentric circles
        outer_radius = int(self.CELL_SIZE * 0.4)
        inner_radius = outer_radius // 2

        pygame.draw.circle(self.screen, (255, 0, 0), (center_x, center_y), outer_radius)  # Red outer ring
        pygame.draw.circle(self.screen, (225, 255, 255), (center_x, center_y), inner_radius)  # White middle ring
        pygame.draw.circle(self.screen, (255, 0, 0), (center_x, center_y), inner_radius // 2)  # Red bull's eye

    def draw_walls(self):
        """
        Render detailed brick-patterned walls at each wall position.
        Uses a sophisticated brick pattern with mortar joints for visual realism.
        """
        brick_color = (139, 69, 19)  # Brown bricks
        mortar_color = (169, 169, 169)  # Gray mortar between bricks
        
        for wall in self.game.wall_positions:
            x, y = self.position_to_grid(wall)
            # Create the base wall with a small inset (2px) from the cell border
            rect = pygame.Rect(x + 2, y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
            pygame.draw.rect(self.screen, mortar_color, rect)

            # Calculate brick and mortar dimensions based on cell size
            mortar_thickness = max(1, min(rect.width, rect.height) // 25)
            brick_height = rect.height // 3
            brick_width = rect.width // 2

            # Draw brick pattern row by row
            for brick_row in range((rect.height // brick_height) + 1):
                y_pos = rect.y + brick_row * brick_height
                # Offset every other row for realistic brick staggering
                x_offset = brick_width // 2 if brick_row % 2 else 0

                # Draw individual bricks in each row
                for brick_col in range(-1, (rect.width // brick_width) + 2):
                    x_pos = rect.x + x_offset + brick_col * brick_width
                    brick = pygame.Rect(
                        x_pos + mortar_thickness,
                        y_pos + mortar_thickness,
                        brick_width - mortar_thickness * 2,
                        brick_height - mortar_thickness * 2
                    )

                    # Clip bricks to stay within the cell boundaries
                    if brick.right > rect.left and brick.left < rect.right and \
                            brick.bottom > rect.top and brick.top < rect.bottom:
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

                        # Draw the brick if it's still visible after clipping
                        if brick.width > 0 and brick.height > 0:
                            pygame.draw.rect(self.screen, brick_color, brick)

    def draw_player(self, position):
        """
        Render the player as a cartoonish robot with a head, eyes, antenna and body.
        The charming design adds personality to the agent navigating the maze.
        """
        # Calculate center point of the cell
        center_x = position[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = position[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        size = self.CELL_SIZE // 3  # Base size for the robot features

        # Draw robot head (blue box)
        head_rect = pygame.Rect(
            center_x - size, center_y - size, size * 2, size * 2
        )
        pygame.draw.rect(self.screen, (100, 100, 180), head_rect)

        # Draw robot eyes (yellow rectangles)
        eye_width = size // 2
        eye_height = size // 3
        eye_y = center_y - size // 2
        left_eye = pygame.Rect(
            center_x - size // 2 - eye_width // 2, eye_y, eye_width, eye_height
        )
        right_eye = pygame.Rect(
            center_x + size // 2 - eye_width // 2, eye_y, eye_width, eye_height
        )
        pygame.draw.rect(self.screen, (255, 255, 0), left_eye)
        pygame.draw.rect(self.screen, (255, 255, 0), right_eye)

        # Draw antenna with red tip
        pygame.draw.line(
            self.screen, (200, 0, 0),
            (center_x, center_y - size),
            (center_x, center_y - size - size // 2), 3
        )
        pygame.draw.circle(
            self.screen, (200, 0, 0),
            (center_x, center_y - size - size // 2), size // 4
        )

        # Draw mouth (simple line) and body (darker blue rectangle)
        pygame.draw.line(
            self.screen, (50, 50, 50),
            (center_x - size // 2, center_y + size // 2),
            (center_x + size // 2, center_y + size // 2), 2
        )
        body_rect = pygame.Rect(
            center_x - size * 0.75, center_y + size, size * 1.5, size
        )
        pygame.draw.rect(self.screen, (70, 70, 150), body_rect)

    def display_end_message(self, message):
        """
        Display a victory message with visual effects when the goal is reached.
        Features a pulsating background and celebratory particle decorations.
        """
        # Calculate pulsating alpha (transparency) value based on time
        pulse_value = abs(math.sin(pygame.time.get_ticks() / 300)) * 255
        alpha = min(200, int(pulse_value))

        # Create a semi-transparent surface for the pulsating message bar
        goal_surface = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        goal_surface.fill((0, 100, 0, alpha))  # Green with variable transparency

        # Render the victory text
        font = pygame.font.SysFont(None, 22)
        text_surf = font.render(message, True, (255, 255, 0))  # Yellow text
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, 20))
        goal_surface.blit(text_surf, text_rect)
        
        # Position the message bar near the bottom of the screen
        self.screen.blit(
            goal_surface,
            (0, self.GRID_SIZE * self.CELL_SIZE - 40)
        )

        # Add decorative particles to enhance the celebration effect
        num_particles = 5
        spacing = self.WIDTH // (num_particles + 1)
        y_pos = self.GRID_SIZE * self.CELL_SIZE - 30
        size = 3
        color = (255, 255, 0)  # Yellow particles to match text

        # Draw evenly spaced particles below the message
        for i in range(1, num_particles + 1):
            x = spacing * i
            pygame.draw.circle(self.screen, color, (x, y_pos), size)

    def refresh(self, obs, reward, done, info, delay: float = 0.1):
        """
        Update the visualization based on the latest game state.
        This is the main rendering function called after each action.
        
        Args:
            obs: Current observation from environment
            reward: Reward from last action
            done: Whether episode has terminated
            info: Additional information dictionary
            delay: Time to wait after drawing (seconds)
        """
        # Format the current action and result for display
        action = info.get('action', "None")
        result = f"Pos: {obs['player_position']}, Reward: {reward}, Action: {action}"

        # Update action history with rolling window effect
        if None in self.action_results:
            self.action_results[self.action_results.index(None)] = result
        else:
            self.action_results.pop(0)  # Remove oldest action
            self.action_results.append(result)  # Add newest action

        # Set up frame timing
        self.fps = 60
        self.clock = pygame.time.Clock()

        # Draw all game elements
        self.screen.fill(self.WHITE)  # Clear screen with background color
        self.draw_grid()  # Draw grid lines
        self.draw_goal_room()  # Draw the goal
        self.draw_walls()  # Draw maze walls
        self.draw_player(self.game.current_state['player_position'])  # Draw the agent

        # Check if goal reached and display victory message if so
        if self.game.is_terminal() == 'goal':
            self.game_ended = True
            self.display_end_message("GOAL REACHED!")

        # Update the display and control timing
        pygame.display.flip()  # Update the full display
        self.clock.tick(self.fps)  # Maintain frame rate
        time.sleep(self.sleeptime)  # Additional delay for visualization


if __name__ == "__main__":
    # Simple demo when file is run directly - initialize and display the maze
    gui = DynamicMazeGUI()
    gui.setup()