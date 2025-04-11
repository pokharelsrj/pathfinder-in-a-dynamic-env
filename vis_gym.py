import pygame
import sys
import time
from mdp_gym import CastleEscapeEnv  # Import the updated CastleEscapeEnv (without guards)
import random
import math

# Initialize the updated environment
game = CastleEscapeEnv()

# Screen configuration
GRID_SIZE = game.grid_size
CELL_SIZE = 500 // GRID_SIZE  # Size of each cell in pixels
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)  # Color for the goal room

# Global variables
screen = None
game_ended = False
action_results = [None, None, None, None, None]

fps = 10
sleeptime = 0.1
clock = None


# Initialize Pygame
def setup(GUI=True):
    global screen
    if GUI:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT + 50))
        pygame.display.set_caption("Dynamic Maze Environment")


# Map room coordinate to pixel grid position
def position_to_grid(position):
    row, col = position
    return col * CELL_SIZE, row * CELL_SIZE


# Draw the grid and the console area
def draw_grid():
    for x in range(0, WIDTH, CELL_SIZE):
        for y in range(0, HEIGHT, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
    # Shade the console area
    rect = pygame.Rect(0, HEIGHT + 100, WIDTH, HEIGHT)
    pygame.draw.rect(screen, GRAY, rect)


# Draw the goal room in yellow with a label
def draw_goal_room():
    x, y = position_to_grid(game.goal_room)
    center_x = x + CELL_SIZE // 2
    center_y = y + CELL_SIZE // 2

    # outer radius
    outer_radius = int(CELL_SIZE * 0.4)
    inner_radius = int(outer_radius * 0.5)
    pygame.draw.circle(screen, (255, 0, 0), (center_x, center_y), outer_radius)

    # inner circle
    pygame.draw.circle(screen, (225, 255, 255), (center_x, center_y), inner_radius)

    pygame.draw.circle(screen, (255, 0, 0), (center_x, center_y), inner_radius // 2)

    # rect = pygame.Rect(x, y, CELL_SIZE - 2, CELL_SIZE - 2)
    # pygame.draw.rect(screen, (255, 255, 0, 128), rect)
    # font = pygame.font.Font(None, 36)
    # label = font.render('Goal', True, BLACK)
    # screen.blit(label, (x + CELL_SIZE // 4 + 1, y + CELL_SIZE // 4 + 1))


# Draw walls on the grid
def draw_walls():
    # colors
    brick_color = (139, 69, 19)
    mortar_color = (169, 169, 169)
    for wall in game.wall_positions:
        x, y = position_to_grid(wall)
        # Draw wall as a slightly inset dark gray rectangle
        rect = pygame.Rect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4)
        pygame.draw.rect(screen, mortar_color, rect)

        mortar_thickness = max(1, min(rect.width, rect.height) // 25)
        brick_height = rect.height // 3
        brick_width = rect.width // 2

        # Draw the brick pattern rows
        for brick_row in range((rect.height // brick_height) + 1):
            y_pos = rect.y + brick_row * brick_height
            # Offset every other row for the brick pattern
            x_offset = brick_width // 2 if brick_row % 2 else 0

            # Draw the bricks in this row
            for brick_col in range(-1, (rect.width // brick_width) + 2):
                x_pos = rect.x + x_offset + brick_col * brick_width

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
                        pygame.draw.rect(screen, brick_color, brick)


# Draw the player as a green circle
def draw_player(position):
    """
    Draw a simple robot character using basic shapes
    """
    # Calculate center position
    center_x = position[1] * CELL_SIZE + CELL_SIZE // 2
    center_y = position[0] * CELL_SIZE + CELL_SIZE // 2

    # Robot size (slightly smaller than the cell)
    size = CELL_SIZE // 3

    # Robot head (square)
    head_color = (100, 100, 180)  # Steel blue
    head_rect = pygame.Rect(
        center_x - size,
        center_y - size,
        size * 2,
        size * 2
    )
    pygame.draw.rect(screen, head_color, head_rect)

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
    pygame.draw.rect(screen, eye_color, left_eye_rect)

    # Right eye
    right_eye_rect = pygame.Rect(
        center_x + size // 2 - eye_width // 2,
        eye_y,
        eye_width,
        eye_height
    )
    pygame.draw.rect(screen, eye_color, right_eye_rect)

    # Antenna
    antenna_color = (200, 0, 0)  # Red
    pygame.draw.line(
        screen,
        antenna_color,
        (center_x, center_y - size),
        (center_x, center_y - size - size // 2),
        3
    )

    # Antenna top
    pygame.draw.circle(
        screen,
        antenna_color,
        (center_x, center_y - size - size // 2),
        size // 4
    )

    # Robot mouth (simple line)
    mouth_y = center_y + size // 2
    pygame.draw.line(
        screen,
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
    pygame.draw.rect(screen, body_color, body_rect)


# Display an end-of-game message (e.g., "Victory!")
def display_end_message(message):
    # Create a pulsating effect
    pulse_value = abs(math.sin(pygame.time.get_ticks() / 300)) * 255
    goal_surface = pygame.Surface((WIDTH, 40), pygame.SRCALPHA)
    goal_surface.fill((0, 100, 0, min(200, int(pulse_value))))

    goal_font = pygame.font.SysFont(None, 22)
    goal_reached_text = goal_font.render(message, True, (255, 255, 0))
    goal_reached_rect = goal_reached_text.get_rect(center=(WIDTH // 2, 20))
    goal_surface.blit(goal_reached_text, goal_reached_rect)

    # Position at the bottom of the grid, above the panel
    screen.blit(goal_surface, (0, GRID_SIZE * CELL_SIZE - 40))

    # Add celebration particles
    if random.random() < 0.3:  # Only create particles occasionally
        for _ in range(5):
            x = random.randint(0, WIDTH)
            y = random.randint(GRID_SIZE * CELL_SIZE - 60, GRID_SIZE * CELL_SIZE - 20)
            size = random.randint(2, 5)
            color = random.choice([(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)])
            pygame.draw.circle(screen, color, (x, y), size)


# Main loop for the Pygame visualization
def main():
    global game_ended, action_results
    clock = pygame.time.Clock()
    running = True
    end_message = ""

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Only allow movement actions now (W, A, S, D)
                if not game_ended:
                    if event.key == pygame.K_w:
                        action = "UP"
                        result = game.step(action)
                        action_results.append(f"Action: {action}, Result: {result}")
                    if event.key == pygame.K_s:
                        action = "DOWN"
                        result = game.step(action)
                        action_results.append(f"Action: {action}, Result: {result}")
                    if event.key == pygame.K_a:
                        action = "LEFT"
                        result = game.step(action)
                        action_results.append(f"Action: {action}, Result: {result}")
                    if event.key == pygame.K_d:
                        action = "RIGHT"
                        result = game.step(action)
                        action_results.append(f"Action: {action}, Result: {result}")
                    if event.key == pygame.K_r:
                        action = "RESET"
                        result = game.reset()
                        action_results.append(f"Action: {action}, Result: {result}")
        screen.fill(WHITE)
        draw_grid()

        # Draw goal room, walls, and player
        draw_goal_room()
        draw_walls()
        draw_player(game.current_state['player_position'])

        # Check for terminal state (goal reached)
        if game.is_terminal() == 'goal':
            game_ended = True
            end_message = "GOAL REACHED!"

        if game_ended:
            display_end_message(end_message)
            # Optionally, you can stop the game loop here instead of resetting game_ended
            game_ended = False

        reward = 0
        # Display reward with a fancy format centered at the top of the panel
        reward_font = pygame.font.SysFont(None, 28)
        reward_text = reward_font.render(f"SCORE: {reward}", True, BLUE)
        reward_rect = reward_text.get_rect(center=(WIDTH // 2, HEIGHT + 25))
        screen.blit(reward_text, reward_rect)

        # Display the latest 5 console messages
        # font = pygame.font.Font(None, 30)
        # console_surface = font.render("Console", True, BLUE)
        # screen.blit(console_surface, (10, HEIGHT + 10))
        # font = pygame.font.Font(None, 24)
        # y_offset = HEIGHT + 45
        # for result in action_results[-5:]:
        #    if result is not None:
        #        result_surface = font.render(result, True, BLACK)
        #        screen.blit(result_surface, (10, y_offset))
        #        y_offset += 30

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()


# Optional refresh function for updating display from external calls
def refresh(obs, reward, done, info, delay=0.1):
    global fps, sleeptime, game_ended, clock, action_results, game

    try:
        action = info['action']
    except:
        action = "None"

    result = "Pos: {}, Reward: {}, Action: {}".format(obs['player_position'], reward, action)

    if None in action_results:
        action_results[action_results.index(None)] = result
    else:
        action_results.pop(0)
        action_results.append(result)

    fps = 60
    clock = pygame.time.Clock()
    screen.fill(WHITE)
    draw_grid()
    draw_goal_room()
    draw_walls()
    draw_player(game.current_state['player_position'])

    if game.is_terminal() == 'goal':
        game_ended = True
        end_message = "GOAL REACHED!"
        display_end_message(end_message)
    pygame.display.flip()
    clock.tick(fps)
    time.sleep(sleeptime)


if __name__ == "__main__":
    setup()
    main()
