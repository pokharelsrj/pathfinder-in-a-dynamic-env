import pygame
import sys
import time
import random
import math
from gui import DynamicMazeEnv  # Import the updated DynamicMazeEnv


class DynamicMazeGUI:
    def __init__(self, gui: bool = True):
        # Game and grid configuration
        self.game = DynamicMazeEnv()
        self.GRID_SIZE = self.game.grid_size
        self.CELL_SIZE = 500 // self.GRID_SIZE  # pixels per cell
        self.WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.HEIGHT = self.GRID_SIZE * self.CELL_SIZE

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (200, 200, 200)
        self.DARK_GRAY = (50, 50, 50)
        self.YELLOW = (255, 255, 0)

        # Pygame screen and state
        self.screen = None
        self.game_ended = False
        self.action_results = [None, None, None, None, None]

        # Timing
        self.fps = 10
        self.sleeptime = 0.1
        self.clock = None

        if gui:
            self.setup()

    def setup(self):
        """Initialize Pygame and create window."""
        pygame.init()
        # Extra 50 pixels for messages
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT + 50))
        pygame.display.set_caption("Dynamic Maze Environment")

    def position_to_grid(self, position):
        """Convert (row, col) to pixel (x, y)."""
        row, col = position
        return col * self.CELL_SIZE, row * self.CELL_SIZE

    def draw_grid(self):
        """Draw grid lines and console shading."""
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            for y in range(0, self.HEIGHT, self.CELL_SIZE):
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)
        # Shade extra console area (unused)
        rect = pygame.Rect(0, self.HEIGHT + 100, self.WIDTH, self.HEIGHT)
        pygame.draw.rect(self.screen, self.GRAY, rect)

    def draw_goal_room(self):
        """Highlight the goal room with concentric circles."""
        x, y = self.position_to_grid(self.game.goal_room)
        center_x = x + self.CELL_SIZE // 2
        center_y = y + self.CELL_SIZE // 2

        outer_radius = int(self.CELL_SIZE * 0.4)
        inner_radius = outer_radius // 2

        pygame.draw.circle(self.screen, (255, 0, 0), (center_x, center_y), outer_radius)
        pygame.draw.circle(self.screen, (225, 255, 255), (center_x, center_y), inner_radius)
        pygame.draw.circle(self.screen, (255, 0, 0), (center_x, center_y), inner_radius // 2)

    def draw_walls(self):
        """Render brick-patterned walls at wall positions."""
        brick_color = (139, 69, 19)
        mortar_color = (169, 169, 169)
        for wall in self.game.wall_positions:
            x, y = self.position_to_grid(wall)
            rect = pygame.Rect(x + 2, y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
            pygame.draw.rect(self.screen, mortar_color, rect)

            mortar_thickness = max(1, min(rect.width, rect.height) // 25)
            brick_height = rect.height // 3
            brick_width = rect.width // 2

            for brick_row in range((rect.height // brick_height) + 1):
                y_pos = rect.y + brick_row * brick_height
                x_offset = brick_width // 2 if brick_row % 2 else 0

                for brick_col in range(-1, (rect.width // brick_width) + 2):
                    x_pos = rect.x + x_offset + brick_col * brick_width
                    brick = pygame.Rect(
                        x_pos + mortar_thickness,
                        y_pos + mortar_thickness,
                        brick_width - mortar_thickness * 2,
                        brick_height - mortar_thickness * 2
                    )

                    # Clip to cell
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

                        if brick.width > 0 and brick.height > 0:
                            pygame.draw.rect(self.screen, brick_color, brick)

    def draw_player(self, position):
        """Render the player as a simple robot figure."""
        center_x = position[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = position[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        size = self.CELL_SIZE // 3

        # Head
        head_rect = pygame.Rect(
            center_x - size, center_y - size, size * 2, size * 2
        )
        pygame.draw.rect(self.screen, (100, 100, 180), head_rect)

        # Eyes
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

        # Antenna
        pygame.draw.line(
            self.screen, (200, 0, 0),
            (center_x, center_y - size),
            (center_x, center_y - size - size // 2), 3
        )
        pygame.draw.circle(
            self.screen, (200, 0, 0),
            (center_x, center_y - size - size // 2), size // 4
        )

        # Mouth and body
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
        """Show pulsating end-game message with particles."""
        pulse_value = abs(math.sin(pygame.time.get_ticks() / 300)) * 255
        goal_surface = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        goal_surface.fill((0, 100, 0, min(200, int(pulse_value))))

        font = pygame.font.SysFont(None, 22)
        text_surf = font.render(message, True, (255, 255, 0))
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, 20))
        goal_surface.blit(text_surf, text_rect)
        self.screen.blit(goal_surface, (0, self.GRID_SIZE * self.CELL_SIZE - 40))

        if random.random() < 0.3:
            for _ in range(5):
                x = random.randint(0, self.WIDTH)
                y = random.randint(self.GRID_SIZE * self.CELL_SIZE - 60,
                                   self.GRID_SIZE * self.CELL_SIZE - 20)
                size = random.randint(2, 5)
                color = random.choice([(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)])
                pygame.draw.circle(self.screen, color, (x, y), size)

    def refresh(self, obs, reward, done, info, delay: float = 0.1):
        """Update visuals based on the latest game state."""
        action = info.get('action', "None")
        result = f"Pos: {obs['player_position']}, Reward: {reward}, Action: {action}"

        # Maintain a rolling action history
        if None in self.action_results:
            self.action_results[self.action_results.index(None)] = result
        else:
            self.action_results.pop(0)
            self.action_results.append(result)

        self.fps = 60
        self.clock = pygame.time.Clock()

        self.screen.fill(self.WHITE)
        self.draw_grid()
        self.draw_goal_room()
        self.draw_walls()
        self.draw_player(self.game.current_state['player_position'])

        if self.game.is_terminal() == 'goal':
            self.game_ended = True
            self.display_end_message("GOAL REACHED!")

        pygame.display.flip()
        self.clock.tick(self.fps)
        time.sleep(self.sleeptime)


if __name__ == "__main__":
    gui = DynamicMazeGUI()
    gui.setup()
