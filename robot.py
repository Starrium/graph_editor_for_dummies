import pygame
import sys
import json
from tkinter import Tk, filedialog
import random
import math

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700
GRID_SIZE = 30  # Number of cells
CELL_SIZE = 20  # Pixel size of each cell
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 100

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
BLUE = (0, 120, 215)
RED = (220, 20, 60)
GREEN = (50, 205, 50)
YELLOW = (255, 215, 0)
ORANGE = (255, 140, 0)
PURPLE = (147, 112, 219)
LIGHT_BLUE = (173, 216, 230)

# GA Parameters
POPULATION_SIZE = 50
MAX_GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
ELITE_SIZE = 5


class GridEnvironment:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        # 0 = free, 1 = obstacle, 2 = start, 3 = goal
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.start_pos = None
        self.goal_pos = None

    def set_cell(self, row, col, value):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            # Clear previous start/goal if setting new one
            if value == 2:  # Start
                if self.start_pos:
                    old_r, old_c = self.start_pos
                    self.grid[old_r][old_c] = 0
                self.start_pos = (row, col)
            elif value == 3:  # Goal
                if self.goal_pos:
                    old_r, old_c = self.goal_pos
                    self.grid[old_r][old_c] = 0
                self.goal_pos = (row, col)

            self.grid[row][col] = value

    def get_cell(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        return None

    def is_valid_pos(self, row, col):
        """Check if position is valid and not an obstacle"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col] != 1
        return False

    def clear_grid(self):
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.start_pos = None
        self.goal_pos = None

    def to_dict(self):
        """Export grid data to dictionary"""
        return {
            'rows': self.rows,
            'cols': self.cols,
            'grid': self.grid,
            'start_pos': self.start_pos,
            'goal_pos': self.goal_pos
        }

    def from_dict(self, data):
        """Import grid data from dictionary"""
        self.rows = data['rows']
        self.cols = data['cols']
        self.grid = data['grid']
        self.start_pos = tuple(data['start_pos']) if data['start_pos'] else None
        self.goal_pos = tuple(data['goal_pos']) if data['goal_pos'] else None


class Chromosome:
    """Represents a path from start to goal"""

    def __init__(self, path):
        self.path = path  # List of (row, col) tuples
        self.fitness = 0.0

    def __len__(self):
        return len(self.path)


class GeneticAlgorithm:
    def __init__(self, environment):
        self.env = environment
        self.population = []
        self.generation = 0
        self.best_chromosome = None
        self.best_fitness = float('-inf')

    def euclidean_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def generate_random_path(self):
        """Generate a random valid path from start to goal using directed walk"""
        if not self.env.start_pos or not self.env.goal_pos:
            return None

        path = [self.env.start_pos]
        current = self.env.start_pos
        max_steps = self.env.rows * self.env.cols

        for _ in range(max_steps):
            if current == self.env.goal_pos:
                break

            # Get all valid neighbors
            neighbors = []
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8-directional

            for dr, dc in directions:
                new_r, new_c = current[0] + dr, current[1] + dc
                if self.env.is_valid_pos(new_r, new_c) and (new_r, new_c) not in path:
                    neighbors.append((new_r, new_c))

            if not neighbors:
                # No valid moves, try to reach goal directly if possible
                if self.env.is_valid_pos(self.env.goal_pos[0], self.env.goal_pos[1]):
                    path.append(self.env.goal_pos)
                break

            # Bias towards goal (70% of the time)
            if random.random() < 0.7:
                # Choose neighbor closest to goal
                neighbors.sort(key=lambda p: self.euclidean_distance(p, self.env.goal_pos))
                current = neighbors[0]
            else:
                # Random choice for diversity
                current = random.choice(neighbors)

            path.append(current)

        # Ensure goal is included
        if path[-1] != self.env.goal_pos:
            path.append(self.env.goal_pos)

        return path

    def initialize_population(self):
        """Create initial population of random paths"""
        self.population = []
        attempts = 0
        max_attempts = POPULATION_SIZE * 10

        while len(self.population) < POPULATION_SIZE and attempts < max_attempts:
            path = self.generate_random_path()
            if path and len(path) > 1:
                chromosome = Chromosome(path)
                self.population.append(chromosome)
            attempts += 1

        # Fill remaining with duplicates if needed
        while len(self.population) < POPULATION_SIZE:
            if self.population:
                self.population.append(Chromosome(self.population[0].path[:]))
            else:
                break

    def calculate_fitness(self, chromosome):
        """Calculate fitness based on distance, safety, and energy (turns)"""
        if len(chromosome.path) < 2:
            return -float('inf')

        # Validate path - ensure no obstacles and all moves are adjacent
        for i in range(len(chromosome.path)):
            pos = chromosome.path[i]
            # Check if position is valid
            if not self.env.is_valid_pos(pos[0], pos[1]):
                return -float('inf')

            # Check if moves are adjacent
            if i > 0:
                prev = chromosome.path[i - 1]
                dr = abs(pos[0] - prev[0])
                dc = abs(pos[1] - prev[1])
                if dr > 1 or dc > 1:
                    return -float('inf')

        # Path length (Euclidean distance)
        path_length = 0
        for i in range(len(chromosome.path) - 1):
            path_length += self.euclidean_distance(chromosome.path[i], chromosome.path[i + 1])

        # Safety: check proximity to obstacles
        safety_penalty_1 = 0  # First level (adjacent)
        safety_penalty_2 = 0  # Second level (distance 2)

        for pos in chromosome.path[1:-1]:  # Exclude start and goal
            # Check first level (8 neighbors)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = pos[0] + dr, pos[1] + dc
                    if 0 <= r < self.env.rows and 0 <= c < self.env.cols:
                        if self.env.grid[r][c] == 1:
                            safety_penalty_1 += 1

            # Check second level (distance 2)
            for dr in [-2, -1, 0, 1, 2]:
                for dc in [-2, -1, 0, 1, 2]:
                    if abs(dr) <= 1 and abs(dc) <= 1:
                        continue
                    r, c = pos[0] + dr, pos[1] + dc
                    if 0 <= r < self.env.rows and 0 <= c < self.env.cols:
                        if self.env.grid[r][c] == 1:
                            safety_penalty_2 += 1

        # Energy: count turns in the path
        turns = 0
        for i in range(1, len(chromosome.path) - 1):
            prev = chromosome.path[i - 1]
            curr = chromosome.path[i]
            next_pos = chromosome.path[i + 1]

            # Calculate direction vectors
            dir1 = (curr[0] - prev[0], curr[1] - prev[1])
            dir2 = (next_pos[0] - curr[0], next_pos[1] - curr[1])

            # If direction changes, it's a turn
            if dir1 != dir2:
                turns += 1

        # Fitness function (higher is better)
        # F = 1 / (w_l * length + w_s1 * safety1 + w_s2 * safety2) - turns
        w_l = 1.0
        w_s1 = 2.0
        w_s2 = 0.5

        denominator = w_l * path_length + w_s1 * safety_penalty_1 + w_s2 * safety_penalty_2
        if denominator == 0:
            denominator = 0.001

        fitness = (1.0 / denominator) - (turns * 0.1)

        return fitness

    def evaluate_population(self):
        """Evaluate fitness for all chromosomes"""
        for chromosome in self.population:
            chromosome.fitness = self.calculate_fitness(chromosome)

        # Sort by fitness (descending)
        self.population.sort(key=lambda c: c.fitness, reverse=True)

        # Update best
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_chromosome = Chromosome(self.population[0].path[:])

    def tournament_selection(self, tournament_size=3):
        """Select parent using tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda c: c.fitness)

    def crossover(self, parent1, parent2):
        """Improved Same Adjacency Crossover"""
        # Find common adjacent positions
        crossover_points = []

        for i in range(len(parent1.path) - 1):
            for j in range(len(parent2.path) - 1):
                p1_curr, p1_next = parent1.path[i], parent1.path[i + 1]
                p2_curr, p2_next = parent2.path[j], parent2.path[j + 1]

                # Check if paths can be crossed at these points
                # Both crossover connections must be feasible
                if (self.is_path_feasible(p1_curr, p2_next) and
                        self.is_path_feasible(p2_curr, p1_next)):
                    crossover_points.append((i, j))

        if not crossover_points:
            # No valid crossover, return copies of parents
            return Chromosome(parent1.path[:]), Chromosome(parent2.path[:])

        # Choose a random crossover point
        i, j = random.choice(crossover_points)

        # Create offspring
        offspring1_path = parent1.path[:i + 1] + parent2.path[j + 1:]
        offspring2_path = parent2.path[:j + 1] + parent1.path[i + 1:]

        # Validate offspring paths - remove invalid ones
        offspring1 = Chromosome(offspring1_path) if self.is_valid_path(offspring1_path) else Chromosome(parent1.path[:])
        offspring2 = Chromosome(offspring2_path) if self.is_valid_path(offspring2_path) else Chromosome(parent2.path[:])

        return offspring1, offspring2

    def is_valid_path(self, path):
        """Validate entire path for obstacles and adjacency"""
        if len(path) < 2:
            return False

        for i in range(len(path)):
            # Check each position is valid
            if not self.env.is_valid_pos(path[i][0], path[i][1]):
                return False

            # Check adjacency between consecutive positions
            if i > 0:
                dr = abs(path[i][0] - path[i - 1][0])
                dc = abs(path[i][1] - path[i - 1][1])
                if dr > 1 or dc > 1:
                    return False

        return True

    def is_path_feasible(self, pos1, pos2):
        """Check if direct path between two positions is feasible"""
        # Positions must be adjacent (8-directional) and both must be valid
        dr = abs(pos1[0] - pos2[0])
        dc = abs(pos1[1] - pos2[1])

        if not ((dr <= 1 and dc <= 1) or pos1 == pos2):
            return False

        # Both positions must not be obstacles
        return self.env.is_valid_pos(pos1[0], pos1[1]) and self.env.is_valid_pos(pos2[0], pos2[1])

    def mutate(self, chromosome):
        """Mutate chromosome by modifying a random segment"""
        if len(chromosome.path) < 3 or random.random() > MUTATION_RATE:
            return

        # Select a random point in the path (not start or end)
        mutation_point = random.randint(1, len(chromosome.path) - 2)

        # Try to replace with a valid neighbor
        current = chromosome.path[mutation_point]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        valid_neighbors = []
        for dr, dc in directions:
            new_r, new_c = current[0] + dr, current[1] + dc
            if self.env.is_valid_pos(new_r, new_c):
                valid_neighbors.append((new_r, new_c))

        if valid_neighbors:
            new_pos = random.choice(valid_neighbors)

            # Only mutate if it maintains path connectivity
            prev_connected = True
            next_connected = True

            if mutation_point > 0:
                prev = chromosome.path[mutation_point - 1]
                dr = abs(new_pos[0] - prev[0])
                dc = abs(new_pos[1] - prev[1])
                prev_connected = (dr <= 1 and dc <= 1)

            if mutation_point < len(chromosome.path) - 1:
                next_pos = chromosome.path[mutation_point + 1]
                dr = abs(new_pos[0] - next_pos[0])
                dc = abs(new_pos[1] - next_pos[1])
                next_connected = (dr <= 1 and dc <= 1)

            if prev_connected and next_connected:
                chromosome.path[mutation_point] = new_pos

    def evolve(self):
        """Evolve population for one generation"""
        new_population = []

        # Elitism: keep best chromosomes
        new_population.extend([Chromosome(c.path[:]) for c in self.population[:ELITE_SIZE]])

        # Generate rest of population through crossover and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            if random.random() < CROSSOVER_RATE:
                offspring1, offspring2 = self.crossover(parent1, parent2)
            else:
                offspring1 = Chromosome(parent1.path[:])
                offspring2 = Chromosome(parent2.path[:])

            self.mutate(offspring1)
            self.mutate(offspring2)

            new_population.append(offspring1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(offspring2)

        self.population = new_population[:POPULATION_SIZE]
        self.generation += 1


class Button:
    def __init__(self, x, y, width, height, text, color, text_color=WHITE):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hovered = False

    def draw(self, screen, font):
        color = tuple(min(c + 30, 255) for c in self.color) if self.hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)

        text_surface = font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

    def update_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)


class RobotSimulator:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Autonomous Robot Path Planning Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 36)

        self.environment = GridEnvironment(GRID_SIZE, GRID_SIZE)
        self.drawing = False
        self.current_mode = 1  # 0=erase, 1=obstacle, 2=start, 3=goal

        self.ga = None
        self.running_ga = False
        self.ga_speed = 2  # Generations per second (lowered from 5)
        self.last_ga_update = 0

        # Create buttons
        button_y = 20
        button_spacing = 110
        self.buttons = {
            'obstacle': Button(50, button_y, 100, 40, 'Obstacle', DARK_GRAY),
            'start': Button(50 + button_spacing, button_y, 100, 40, 'Start', GREEN),
            'goal': Button(50 + button_spacing * 2, button_y, 100, 40, 'Goal', RED),
            'erase': Button(50 + button_spacing * 3, button_y, 100, 40, 'Erase', YELLOW, BLACK),
            'clear': Button(50 + button_spacing * 4, button_y, 100, 40, 'Clear All', BLUE),
            'run': Button(50 + button_spacing * 5, button_y, 100, 40, 'Run GA', GREEN),
            'export': Button(50 + button_spacing * 6, button_y, 100, 40, 'Export', (150, 100, 200)),
            'import': Button(50 + button_spacing * 7, button_y, 100, 40, 'Import', (200, 150, 100)),
            'speed_down': Button(50 + button_spacing * 8, button_y, 45, 40, '<<', PURPLE),
            'speed_up': Button(50 + button_spacing * 8 + 50, button_y, 45, 40, '>>', PURPLE),
        }

        # Initialize Tkinter for file dialogs (hidden window)
        self.tk_root = Tk()
        self.tk_root.withdraw()

    def get_grid_pos(self, mouse_pos):
        x, y = mouse_pos
        col = (x - GRID_OFFSET_X) // CELL_SIZE
        row = (y - GRID_OFFSET_Y) // CELL_SIZE

        if 0 <= row < self.environment.rows and 0 <= col < self.environment.cols:
            return row, col
        return None

    def draw_grid(self):
        # Draw grid cells
        for row in range(self.environment.rows):
            for col in range(self.environment.cols):
                x = GRID_OFFSET_X + col * CELL_SIZE
                y = GRID_OFFSET_Y + row * CELL_SIZE

                cell_value = self.environment.get_cell(row, col)

                # Determine cell color
                if cell_value == 0:  # Free
                    color = WHITE
                elif cell_value == 1:  # Obstacle
                    color = DARK_GRAY
                elif cell_value == 2:  # Start
                    color = GREEN
                elif cell_value == 3:  # Goal
                    color = RED

                pygame.draw.rect(self.screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE), 1)

    def draw_paths(self):
        """Draw current population paths"""
        if not self.ga or not self.ga.population:
            return

        # Draw all paths in population (semi-transparent)
        for i, chromosome in enumerate(self.ga.population[:10]):  # Show top 10
            if len(chromosome.path) < 2:
                continue

            # Color gradient from blue to purple
            alpha = 30 + (i * 20)
            color = (100, 100, 255 - (i * 10))

            for j in range(len(chromosome.path) - 1):
                start_pos = chromosome.path[j]
                end_pos = chromosome.path[j + 1]

                start_x = GRID_OFFSET_X + start_pos[1] * CELL_SIZE + CELL_SIZE // 2
                start_y = GRID_OFFSET_Y + start_pos[0] * CELL_SIZE + CELL_SIZE // 2
                end_x = GRID_OFFSET_X + end_pos[1] * CELL_SIZE + CELL_SIZE // 2
                end_y = GRID_OFFSET_Y + end_pos[0] * CELL_SIZE + CELL_SIZE // 2

                pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), 1)

        # Draw best path (thick and bright)
        if self.ga.best_chromosome and len(self.ga.best_chromosome.path) >= 2:
            for i in range(len(self.ga.best_chromosome.path) - 1):
                start_pos = self.ga.best_chromosome.path[i]
                end_pos = self.ga.best_chromosome.path[i + 1]

                start_x = GRID_OFFSET_X + start_pos[1] * CELL_SIZE + CELL_SIZE // 2
                start_y = GRID_OFFSET_Y + start_pos[0] * CELL_SIZE + CELL_SIZE // 2
                end_x = GRID_OFFSET_X + end_pos[1] * CELL_SIZE + CELL_SIZE // 2
                end_y = GRID_OFFSET_Y + end_pos[0] * CELL_SIZE + CELL_SIZE // 2

                pygame.draw.line(self.screen, ORANGE, (start_x, start_y), (end_x, end_y), 3)

    def draw_ui(self):
        # Draw buttons
        for button in self.buttons.values():
            button.draw(self.screen, self.font)

        # Highlight selected mode
        if self.current_mode == 0:
            pygame.draw.rect(self.screen, BLUE, self.buttons['erase'].rect, 3)
        elif self.current_mode == 1:
            pygame.draw.rect(self.screen, BLUE, self.buttons['obstacle'].rect, 3)
        elif self.current_mode == 2:
            pygame.draw.rect(self.screen, BLUE, self.buttons['start'].rect, 3)
        elif self.current_mode == 3:
            pygame.draw.rect(self.screen, BLUE, self.buttons['goal'].rect, 3)

        # Draw instructions
        instructions = [
            "Click and drag to draw obstacles | Export/Import to save/load | Use << >> to control speed",
            "Set start (green) and goal (red) positions",
            "Click 'Run GA' to find the optimal path"
        ]

        y_offset = GRID_OFFSET_Y + self.environment.rows * CELL_SIZE + 20
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, BLACK)
            self.screen.blit(text, (GRID_OFFSET_X, y_offset + i * 20))

    def draw_ga_stats(self):
        """Draw GA statistics"""
        if not self.ga:
            return

        stats_x = GRID_OFFSET_X + self.environment.cols * CELL_SIZE + 30
        stats_y = GRID_OFFSET_Y

        # Title
        title = self.font.render("GA Statistics", True, BLACK)
        self.screen.blit(title, (stats_x, stats_y))

        # Stats
        stats = [
            f"Generation: {self.ga.generation}/{MAX_GENERATIONS}",
            f"Population: {len(self.ga.population)}",
            f"Best Fitness: {self.ga.best_fitness:.4f}",
            "",
        ]

        if self.ga.best_chromosome:
            turns = self.count_turns(self.ga.best_chromosome.path)
            path_length = self.calculate_path_length(self.ga.best_chromosome.path)
            stats.extend([
                f"Best Path:",
                f"  Length: {path_length:.2f}",
                f"  Nodes: {len(self.ga.best_chromosome.path)}",
                f"  Turns: {turns}",
            ])

        for i, stat in enumerate(stats):
            text = self.small_font.render(stat, True, BLACK)
            self.screen.blit(text, (stats_x, stats_y + 30 + i * 20))

        # Speed control
        speed_y = stats_y + 30 + len(stats) * 20 + 10
        speed_text = self.small_font.render(f"Speed: {self.ga_speed} gen/sec", True, BLACK)
        self.screen.blit(speed_text, (stats_x, speed_y))

        # Status
        status_y = speed_y + 30
        if self.running_ga:
            status_text = "Status: Running..."
            status_color = GREEN
        elif self.ga.generation >= MAX_GENERATIONS:
            status_text = "Status: Complete!"
            status_color = BLUE
        else:
            status_text = "Status: Ready"
            status_color = BLACK

        status = self.font.render(status_text, True, status_color)
        self.screen.blit(status, (stats_x, status_y))

    def count_turns(self, path):
        """Count number of turns in path"""
        if len(path) < 3:
            return 0

        turns = 0
        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            next_pos = path[i + 1]

            dir1 = (curr[0] - prev[0], curr[1] - prev[1])
            dir2 = (next_pos[0] - curr[0], next_pos[1] - curr[1])

            if dir1 != dir2:
                turns += 1

        return turns

    def calculate_path_length(self, path):
        """Calculate total Euclidean path length"""
        length = 0
        for i in range(len(path) - 1):
            dx = path[i + 1][1] - path[i][1]
            dy = path[i + 1][0] - path[i][0]
            length += math.sqrt(dx * dx + dy * dy)
        return length

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = pygame.mouse.get_pos()

                    # Check button clicks
                    if self.buttons['obstacle'].is_clicked(mouse_pos):
                        self.current_mode = 1
                    elif self.buttons['start'].is_clicked(mouse_pos):
                        self.current_mode = 2
                    elif self.buttons['goal'].is_clicked(mouse_pos):
                        self.current_mode = 3
                    elif self.buttons['erase'].is_clicked(mouse_pos):
                        self.current_mode = 0
                    elif self.buttons['clear'].is_clicked(mouse_pos):
                        self.environment.clear_grid()
                        self.ga = None
                        self.running_ga = False
                    elif self.buttons['run'].is_clicked(mouse_pos):
                        self.run_genetic_algorithm()
                    elif self.buttons['export'].is_clicked(mouse_pos):
                        self.export_environment()
                    elif self.buttons['import'].is_clicked(mouse_pos):
                        self.import_environment()
                    elif self.buttons['speed_down'].is_clicked(mouse_pos):
                        self.ga_speed = max(1, self.ga_speed - 1)
                        print(f"Speed: {self.ga_speed} generations/sec")
                    elif self.buttons['speed_up'].is_clicked(mouse_pos):
                        self.ga_speed = min(20, self.ga_speed + 1)
                        print(f"Speed: {self.ga_speed} generations/sec")
                    else:
                        # Start drawing on grid
                        grid_pos = self.get_grid_pos(mouse_pos)
                        if grid_pos and not self.running_ga:
                            self.drawing = True
                            if self.current_mode == 0:  # Erase mode
                                self.environment.set_cell(grid_pos[0], grid_pos[1], 0)
                            else:
                                self.environment.set_cell(grid_pos[0], grid_pos[1], self.current_mode)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.drawing = False

            elif event.type == pygame.MOUSEMOTION:
                # Update button hover states
                mouse_pos = pygame.mouse.get_pos()
                for button in self.buttons.values():
                    button.update_hover(mouse_pos)

                # Continue drawing if mouse is held down
                if self.drawing and not self.running_ga:
                    grid_pos = self.get_grid_pos(mouse_pos)
                    if grid_pos:
                        if self.current_mode == 0:  # Erase mode
                            self.environment.set_cell(grid_pos[0], grid_pos[1], 0)
                        elif self.current_mode == 1:  # Only allow dragging for obstacles and erase
                            self.environment.set_cell(grid_pos[0], grid_pos[1], self.current_mode)

        return True

    def run_genetic_algorithm(self):
        if not self.environment.start_pos or not self.environment.goal_pos:
            print("Please set both start and goal positions!")
            return

        if self.running_ga:
            # Stop GA
            self.running_ga = False
            self.buttons['run'].text = 'Run GA'
            self.buttons['run'].color = GREEN
        else:
            # Start GA
            print("Starting Genetic Algorithm...")
            self.ga = GeneticAlgorithm(self.environment)
            self.ga.initialize_population()
            self.ga.evaluate_population()
            self.running_ga = True
            self.buttons['run'].text = 'Stop GA'
            self.buttons['run'].color = RED

    def update_ga(self):
        """Update GA state"""
        if not self.running_ga or not self.ga:
            return

        current_time = pygame.time.get_ticks()
        if current_time - self.last_ga_update > 1000 / self.ga_speed:
            if self.ga.generation < MAX_GENERATIONS:
                self.ga.evolve()
                self.ga.evaluate_population()
                self.last_ga_update = current_time
            else:
                self.running_ga = False
                self.buttons['run'].text = 'Run GA'
                self.buttons['run'].color = GREEN
                print(f"GA Complete! Best fitness: {self.ga.best_fitness:.4f}")

    def export_environment(self):
        """Export the current environment to a JSON file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Export Environment"
            )

            if filename:
                data = self.environment.to_dict()
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Environment exported to {filename}")
        except Exception as e:
            print(f"Error exporting environment: {e}")

    def import_environment(self):
        """Import an environment from a JSON file"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Import Environment"
            )

            if filename:
                with open(filename, 'r') as f:
                    data = json.load(f)

                # Validate data
                if 'grid' in data and 'rows' in data and 'cols' in data:
                    self.environment.from_dict(data)
                    self.ga = None
                    self.running_ga = False
                    print(f"Environment imported from {filename}")
                else:
                    print("Invalid environment file format")
        except Exception as e:
            print(f"Error importing environment: {e}")

    def run(self):
        running = True
        while running:
            running = self.handle_events()

            # Update GA if running
            self.update_ga()

            # Draw everything
            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_paths()
            self.draw_ui()
            self.draw_ga_stats()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    simulator = RobotSimulator()
    simulator.run()