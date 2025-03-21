'''
RULES:-

Player Movement:
Players can move using arrow keys (← ↑ → ↓)

Environment:
Hard walls block movement
Soft walls teleport the player to a random adjacent cell and reduce health by 10%
Dynamic Obstacles: Every 3 moves, a new set of obstacles is generated
Game Ending Conditions:
If health reaches 0%, the game ends.
If the player reaches the goal, they win.
'''
import pygame
import random

# Initialize pygame
pygame.init()

# Grid and Display Config
GRID_SIZE = 50
CELL_SIZE = 15
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
FPS = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)

# Game Constants
START_POS = (0, 0)
GOAL_POS = (49, 49)
HARD_OBSTACLE = 1
SOFT_OBSTACLE = 2
OBSTACLE_RATIO = 0.55  # 55% of the grid

# Initialize Screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dynamic Grid Pathfinder")

# Player State
player_pos = list(START_POS)
player_health = 100
move_count = 0

# Generate Initial Obstacles
def generate_obstacles():
    obstacles = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    total_obstacles = int(GRID_SIZE * GRID_SIZE * OBSTACLE_RATIO)
    
    placed = 0
    while placed < total_obstacles:
        x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        if (x, y) in [START_POS, GOAL_POS]:  
            continue  
        if obstacles[x][y] == 0:
            obstacles[x][y] = random.choice([HARD_OBSTACLE, SOFT_OBSTACLE])
            placed += 1
    return obstacles

obstacles = generate_obstacles()

# Draw the Grid
def draw_grid():
    screen.fill(WHITE)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if (x, y) == tuple(player_pos):
                pygame.draw.rect(screen, BLUE, rect)  # Player
            elif (x, y) == GOAL_POS:
                pygame.draw.rect(screen, GREEN, rect)  # Goal
            elif obstacles[x][y] == HARD_OBSTACLE:
                pygame.draw.rect(screen, BLACK, rect)  # Hard Wall
            elif obstacles[x][y] == SOFT_OBSTACLE:
                pygame.draw.rect(screen, RED, rect)  # Soft Wall
            pygame.draw.rect(screen, BLACK, rect, 1)

    # Display Health, Uncomment if need to add extra complexity  
    '''font = pygame.font.Font(None, 30)
    health_text = font.render(f"Health: {player_health}%", True, BLUE)
    screen.blit(health_text, (10, HEIGHT - 30))'''

# Handle Movement
def move_player(dx, dy):
    global move_count, player_health, obstacles

    new_x, new_y = player_pos[0] + dx, player_pos[1] + dy
    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
        if obstacles[new_x][new_y] == HARD_OBSTACLE:
            return  
        elif obstacles[new_x][new_y] == SOFT_OBSTACLE:
            player_health -= 10  
            if player_health <= 0:
                return "Game Over"
            adj_cells = [(new_x + dx, new_y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
            adj_cells = [(x, y) for x, y in adj_cells if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE]
            if adj_cells:
                player_pos[:] = random.choice(adj_cells)
        else:
            player_pos[:] = [new_x, new_y]

    move_count += 1
    if move_count % 3 == 0:
        obstacles = generate_obstacles()  

# Game Loop
running = True
while running:
    pygame.time.delay(100)
    draw_grid()
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                status = move_player(-1, 0)
            elif event.key == pygame.K_RIGHT:
                status = move_player(1, 0)
            elif event.key == pygame.K_UP:
                status = move_player(0, -1)
            elif event.key == pygame.K_DOWN:
                status = move_player(0, 1)

            if player_health <= 0:
                print("Game Over, Better Luck Next Time :)")
                running = False
            elif tuple(player_pos) == GOAL_POS:
                print("Congratulations, You reached the goal :P")
                running = False

pygame.quit()
