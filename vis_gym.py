import pygame
import sys
import time
from mdp_gym import CastleEscapeEnv  # Import the updated CastleEscapeEnv (without guards)

# Initialize the updated environment
game = CastleEscapeEnv()

# Screen configuration
WIDTH, HEIGHT = 600, 840  # 5x5 grid, each room is 120x120 pixels; extra space for console output
GRID_SIZE = game.grid_size
CELL_SIZE = WIDTH // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)  # Color for the goal room

## File paths for images can be added here if needed in the future
IMGFILEPATH = {
    # 'player': 'path/to/player.png',
    # 'wall': 'path/to/wall.png',
    # 'goal': 'path/to/goal.png'
}

# Global variables
screen = None
game_ended = False
action_results = [None, None, None, None, None]

fps = 10
sleeptime = 0
clock = None


# Initialize Pygame
def setup(GUI=True):
    global screen
    if GUI:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Castle Escape MDP Visualization")


# Map room coordinate to pixel grid position
def position_to_grid(position):
    row, col = position
    return col * CELL_SIZE, row * CELL_SIZE


# Draw the grid and the console area
def draw_grid():
    for x in range(0, WIDTH, CELL_SIZE):
        for y in range(0, 600, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
    # Shade the console area
    rect = pygame.Rect(0, 600, WIDTH, HEIGHT - 600)
    pygame.draw.rect(screen, GRAY, rect)


# Draw the goal room in yellow with a label
def draw_goal_room():
    x, y = position_to_grid(game.goal_room)
    rect = pygame.Rect(x, y, CELL_SIZE - 2, CELL_SIZE - 2)
    pygame.draw.rect(screen, YELLOW, rect)
    font = pygame.font.Font(None, 36)
    label = font.render('Goal', True, BLACK)
    screen.blit(label, (x + CELL_SIZE // 4 + 1, y + CELL_SIZE // 4 + 1))


# Draw walls on the grid
def draw_walls():
    for wall in game.wall_positions:
        x, y = position_to_grid(wall)
        # Draw wall as a slightly inset dark gray rectangle
        rect = pygame.Rect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4)
        pygame.draw.rect(screen, DARK_GRAY, rect)


# Draw the player as a green circle
def draw_player(position):
    x, y = position_to_grid(position)
    center_x = x + CELL_SIZE // 2
    center_y = y + CELL_SIZE // 2
    pygame.draw.circle(screen, GREEN, (center_x, center_y), CELL_SIZE // 4)


# Display an end-of-game message (e.g., "Victory!")
def display_end_message(message):
    font = pygame.font.Font(None, 100)
    text_surface = font.render(message, True, DARK_GRAY)
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text_surface, text_rect)


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
        screen.fill(WHITE)
        draw_grid()

        # Draw goal room, walls, and player
        draw_goal_room()
        draw_walls()
        draw_player(game.current_state['player_position'])

        # Check for terminal state (goal reached)
        if game.is_terminal() == 'goal':
            game_ended = True
            end_message = "Victory!"

        if game_ended:
            display_end_message(end_message)
            # Optionally, you can stop the game loop here instead of resetting game_ended
            game_ended = False

        # Display the latest 5 console messages
        font = pygame.font.Font(None, 30)
        console_surface = font.render("Console", True, BLUE)
        screen.blit(console_surface, (10, 610))
        font = pygame.font.Font(None, 24)
        y_offset = 645
        for result in action_results[-5:]:
            if result is not None:
                result_surface = font.render(result, True, BLACK)
                screen.blit(result_surface, (10, y_offset))
                y_offset += 30

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
        end_message = "Victory!"
        display_end_message(end_message)
    pygame.display.flip()
    clock.tick(fps)
    time.sleep(sleeptime)


if __name__ == "__main__":
    setup()
    main()
