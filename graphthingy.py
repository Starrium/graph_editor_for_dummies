import pygame
import sys
import json
import tkinter as tk
from tkinter import filedialog
from collections import deque
import math

pygame.init()
WIDTH, HEIGHT = 1000, 600  # Increased width for side panel
GRAPH_WIDTH = 700  # Width reserved for graph
PANEL_WIDTH = 300  # Width for stack/queue panel
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("totally beginner friendly graph editor")
clock = pygame.time.Clock()

nodes = {}  # {id: (x, y)}
edges = []  # [(u, v)]
node_id = 1  # latest node id
free_ids = set()  # set of usable id
selected_node = None
dragging_node = None
delete_edge_first = None
is_directed = False  # Toggle for directed/undirected graph
RADIUS = 20
FONT = pygame.font.SysFont(None, 24)
BUTTON_FONT = pygame.font.SysFont(None, 22)
SMALL_FONT = pygame.font.SysFont(None, 18)

# setup UI buttons
export_button = pygame.Rect(10, 10, 80, 30)
import_button = pygame.Rect(100, 10, 80, 30)
dfs_button = pygame.Rect(190, 10, 80, 30)
bfs_button = pygame.Rect(280, 10, 80, 30)
step_button = pygame.Rect(370, 10, 80, 30)
reset_button = pygame.Rect(460, 10, 80, 30)
autoplay_button = pygame.Rect(550, 10, 100, 30)
toggle_button = pygame.Rect(10, 50, 110, 30)
clear_button = pygame.Rect(GRAPH_WIDTH - 90, HEIGHT - 40, 80, 30)

# animation status
algo_running = None
algo_state = []
algo_data_structure_states = []  # Store stack/queue state at each step
algo_index = 0
algo_paused = True
autoplay_active = False
animation_speed = 10  # frames per step (lower = faster)
animation_counter = 0
start_node = None


# export function
def save_graph():
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
        title="Save Graph As..."
    )
    root.destroy()
    if not filename:
        return
    data = {
        "nodes": nodes,
        "edges": edges,
        "node_id": node_id,
        "free_ids": list(free_ids),
        "is_directed": is_directed
    }
    with open(filename, "w") as f:
        json.dump(data, f)
    print("exported to: ", filename)


# import function
def load_graph():
    global nodes, edges, node_id, free_ids, is_directed
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        filetypes=[("JSON files", "*.json")],
        title="Open Graph File"
    )
    root.destroy()
    if not filename:
        return
    with open(filename, "r") as f:
        data = json.load(f)
        nodes = {int(k): tuple(v) for k, v in data["nodes"].items()}
        edges = [tuple(e) for e in data["edges"]]
        node_id = data.get("node_id", max(nodes.keys(), default=0) + 1)
        free_ids = set(data.get("free_ids", []))
        is_directed = data.get("is_directed", False)
    print("imported from: ", filename)


def get_node_at(pos):
    mx, my = pos
    for n, (x, y) in nodes.items():
        if mx < GRAPH_WIDTH and (mx - x) ** 2 + (my - y) ** 2 < RADIUS ** 2:
            return n
    return None


def draw_button(rect, text, active=False):
    color = (100, 200, 100) if active else (180, 180, 180)
    pygame.draw.rect(screen, color, rect)
    pygame.draw.rect(screen, (50, 50, 50), rect, 2)
    label = BUTTON_FONT.render(text, True, (0, 0, 0))
    screen.blit(label, (rect.x + (rect.width - label.get_width()) // 2,
                        rect.y + (rect.height - label.get_height()) // 2))


def draw_arrow(surface, color, start, end, width=2):
    """Draw an arrow from start to end position"""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.sqrt(dx ** 2 + dy ** 2)

    if distance == 0:
        return

    dx /= distance
    dy /= distance

    end_x = end[0] - dx * RADIUS
    end_y = end[1] - dy * RADIUS
    start_x = start[0] + dx * RADIUS
    start_y = start[1] + dy * RADIUS

    pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), width)

    arrow_size = 12
    angle = math.atan2(dy, dx)

    left_x = end_x - arrow_size * math.cos(angle - math.pi / 6)
    left_y = end_y - arrow_size * math.sin(angle - math.pi / 6)
    right_x = end_x - arrow_size * math.cos(angle + math.pi / 6)
    right_y = end_y - arrow_size * math.sin(angle + math.pi / 6)

    pygame.draw.polygon(surface, color, [(end_x, end_y), (left_x, left_y), (right_x, right_y)])


def get_neighbors(node):
    neighbors = []
    for u, v in edges:
        if u == node:
            neighbors.append(v)
        elif not is_directed and v == node:
            neighbors.append(u)
    return neighbors


def dfs(start):
    visited = []
    stack = [(start, None)]
    seen = set()
    ds_states = []  # Data structure states

    # Initial state
    ds_states.append(list(stack))

    while stack:
        node, edge = stack.pop()
        if node not in seen:
            seen.add(node)
            visited.append((node, edge))
            for neighbor in get_neighbors(node):
                if neighbor not in seen:
                    stack.append((neighbor, (node, neighbor)))
        # Capture state after each operation
        ds_states.append(list(stack))

    return visited, ds_states


def bfs(start):
    visited = []
    queue = deque([(start, None)])
    seen = set([start])
    ds_states = []  # Data structure states

    # Initial state
    ds_states.append(list(queue))

    while queue:
        node, edge = queue.popleft()
        visited.append((node, edge))
        for neighbor in get_neighbors(node):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, (node, neighbor)))
        # Capture state after each operation
        ds_states.append(list(queue))

    return visited, ds_states


def draw_data_structure_panel():
    """Draw the stack/queue visualization panel on the right side"""
    panel_x = GRAPH_WIDTH

    # Draw panel background
    pygame.draw.rect(screen, (40, 40, 40), (panel_x, 0, PANEL_WIDTH, HEIGHT))
    pygame.draw.line(screen, (100, 100, 100), (panel_x, 0), (panel_x, HEIGHT), 2)

    if not algo_running or algo_index == 0:
        return

    # Get current data structure state
    current_state = algo_data_structure_states[min(algo_index, len(algo_data_structure_states) - 1)]

    # Draw title
    title = "Stack" if algo_running == "DFS" else "Queue"
    title_label = FONT.render(title, True, (255, 255, 255))
    screen.blit(title_label, (panel_x + 10, 10))

    # Draw data structure visualization
    start_y = 50
    item_height = 35
    item_width = PANEL_WIDTH - 40

    if algo_running == "DFS":
        # Draw stack (top to bottom, last element at top)
        stack_label = SMALL_FONT.render("Top", True, (150, 150, 150))
        screen.blit(stack_label, (panel_x + 15, start_y))

        y_offset = start_y + 25
        for i in range(len(current_state) - 1, -1, -1):
            node, edge = current_state[i]

            # Draw stack item box
            rect = pygame.Rect(panel_x + 20, y_offset, item_width, item_height)

            # Color code: lighter at top
            color_intensity = 60 + (len(current_state) - i) * 15
            color_intensity = min(color_intensity, 150)
            pygame.draw.rect(screen, (color_intensity, color_intensity, 100), rect)
            pygame.draw.rect(screen, (200, 200, 200), rect, 2)

            # Draw node label
            node_text = f"Node: {node}"
            node_label = SMALL_FONT.render(node_text, True, (255, 255, 255))
            screen.blit(node_label, (rect.x + 10, rect.y + 8))

            y_offset += item_height + 5

            if y_offset > HEIGHT - 60:
                overflow_text = f"... +{len(current_state) - (len(current_state) - i)} more"
                overflow_label = SMALL_FONT.render(overflow_text, True, (200, 200, 200))
                screen.blit(overflow_label, (panel_x + 20, y_offset))
                break

        if len(current_state) > 0:
            bottom_label = SMALL_FONT.render("Bottom", True, (150, 150, 150))
            screen.blit(bottom_label, (panel_x + 15, min(y_offset, HEIGHT - 45)))

    else:  # BFS - Queue
        # Draw queue (front to back)
        front_label = SMALL_FONT.render("Front", True, (150, 150, 150))
        screen.blit(front_label, (panel_x + 15, start_y))

        y_offset = start_y + 25
        for i, (node, edge) in enumerate(current_state):
            # Draw queue item box
            rect = pygame.Rect(panel_x + 20, y_offset, item_width, item_height)

            # Color code: lighter at front
            color_intensity = 150 - i * 15
            color_intensity = max(color_intensity, 60)
            pygame.draw.rect(screen, (100, color_intensity, color_intensity), rect)
            pygame.draw.rect(screen, (200, 200, 200), rect, 2)

            # Draw node label
            node_text = f"Node: {node}"
            node_label = SMALL_FONT.render(node_text, True, (255, 255, 255))
            screen.blit(node_label, (rect.x + 10, rect.y + 8))

            y_offset += item_height + 5

            if y_offset > HEIGHT - 60:
                overflow_text = f"... +{len(current_state) - i - 1} more"
                overflow_label = SMALL_FONT.render(overflow_text, True, (200, 200, 200))
                screen.blit(overflow_label, (panel_x + 20, y_offset))
                break

        if len(current_state) > 0:
            back_label = SMALL_FONT.render("Back", True, (150, 150, 150))
            screen.blit(back_label, (panel_x + 15, min(y_offset, HEIGHT - 45)))

    # Draw size info
    size_text = f"Size: {len(current_state)}"
    size_label = SMALL_FONT.render(size_text, True, (200, 200, 200))
    screen.blit(size_label, (panel_x + PANEL_WIDTH - 80, 15))


def reset_algo_state():
    global algo_running, algo_state, algo_data_structure_states, algo_index, algo_paused, start_node, autoplay_active, animation_counter
    algo_running = None
    algo_state = []
    algo_data_structure_states = []
    algo_index = 0
    algo_paused = True
    autoplay_active = False
    animation_counter = 0
    start_node = None


def clear_all():
    global nodes, edges, node_id, free_ids, selected_node, delete_edge_first, dragging_node
    nodes = {}
    edges = []
    node_id = 1
    free_ids = set()
    selected_node = None
    delete_edge_first = None
    dragging_node = None
    reset_algo_state()


# main
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                if export_button.collidepoint(event.pos):
                    save_graph()
                    continue
                elif import_button.collidepoint(event.pos):
                    load_graph()
                    continue
                elif toggle_button.collidepoint(event.pos):
                    if not algo_running:
                        is_directed = not is_directed
                    continue
                elif dfs_button.collidepoint(event.pos):
                    if nodes and not algo_running:
                        start_node = min(nodes.keys())
                        algo_running = "DFS"
                        algo_state, algo_data_structure_states = dfs(start_node)
                        algo_index = 0
                        algo_paused = True
                    continue
                elif bfs_button.collidepoint(event.pos):
                    if nodes and not algo_running:
                        start_node = min(nodes.keys())
                        algo_running = "BFS"
                        algo_state, algo_data_structure_states = bfs(start_node)
                        algo_index = 0
                        algo_paused = True
                    continue
                elif step_button.collidepoint(event.pos):
                    if algo_running and algo_index < len(algo_state):
                        algo_index += 1
                        autoplay_active = False
                    continue
                elif autoplay_button.collidepoint(event.pos):
                    if algo_running:
                        autoplay_active = not autoplay_active
                    continue
                elif reset_button.collidepoint(event.pos):
                    reset_algo_state()
                    continue
                elif clear_button.collidepoint(event.pos):
                    clear_all()
                    continue

                # Graph interaction
                if not algo_running and event.pos[0] < GRAPH_WIDTH:
                    node = get_node_at(event.pos)
                    mods = pygame.key.get_mods()

                    if node is None:
                        if free_ids:
                            new_id = min(free_ids)
                            free_ids.remove(new_id)
                        else:
                            new_id = node_id
                            node_id += 1
                        nodes[new_id] = event.pos
                    elif mods & pygame.KMOD_CTRL:
                        if delete_edge_first is None:
                            delete_edge_first = node
                        else:
                            if node != delete_edge_first:
                                edge = (delete_edge_first, node)
                                rev_edge = (node, delete_edge_first)
                                if edge in edges:
                                    edges.remove(edge)
                                elif rev_edge in edges:
                                    edges.remove(rev_edge)
                            delete_edge_first = None
                    elif mods & pygame.KMOD_SHIFT:
                        dragging_node = node
                    else:
                        if selected_node is None:
                            selected_node = node
                        else:
                            if node != selected_node:
                                edge = (selected_node, node)
                                if is_directed:
                                    if edge not in edges:
                                        edges.append(edge)
                                else:
                                    rev_edge = (node, selected_node)
                                    if edge not in edges and rev_edge not in edges:
                                        edges.append(edge)
                            selected_node = None

            elif event.button == 3:  # Right click to delete nodes
                if not algo_running and event.pos[0] < GRAPH_WIDTH:
                    node = get_node_at(event.pos)
                    if node is not None:
                        edges = [(u, v) for (u, v) in edges if u != node and v != node]
                        nodes.pop(node)
                        free_ids.add(node)
                        if selected_node == node:
                            selected_node = None
                        if delete_edge_first == node:
                            delete_edge_first = None

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging_node = None

        elif event.type == pygame.MOUSEMOTION:
            if dragging_node is not None and not algo_running:
                if event.pos[0] < GRAPH_WIDTH:
                    nodes[dragging_node] = event.pos

    # Autoplay logic
    if autoplay_active and algo_running and algo_index < len(algo_state):
        animation_counter += 1
        if animation_counter >= animation_speed:
            algo_index += 1
            animation_counter = 0
        if algo_index >= len(algo_state):
            autoplay_active = False

    # draw
    screen.fill((30, 30, 30))

    # draw buttons
    draw_button(export_button, "Export")
    draw_button(import_button, "Import")
    draw_button(dfs_button, "DFS")
    draw_button(bfs_button, "BFS")
    draw_button(step_button, "Step")
    draw_button(autoplay_button, "Autoplay", autoplay_active)
    draw_button(reset_button, "Reset")
    draw_button(toggle_button, "Directed" if is_directed else "Undirected", is_directed)
    draw_button(clear_button, "Clear All")

    # draw edges
    for u, v in edges:
        color = (200, 200, 200)
        if algo_running and algo_index > 0:
            for i in range(min(algo_index, len(algo_state))):
                node, edge = algo_state[i]
                if edge == (u, v) or (not is_directed and edge == (v, u)):
                    color = (255, 100, 100)
                    break

        if is_directed:
            draw_arrow(screen, color, nodes[u], nodes[v])
        else:
            pygame.draw.line(screen, color, nodes[u], nodes[v], 2)

    # draw nodes
    for n, (x, y) in nodes.items():
        color = (100, 200, 255)
        if n == selected_node:
            color = (100, 255, 100)
        elif n == delete_edge_first:
            color = (255, 180, 50)
        elif algo_running and algo_index > 0:
            for i in range(min(algo_index, len(algo_state))):
                if algo_state[i][0] == n:
                    color = (255, 255, 100)
                    break
        pygame.draw.circle(screen, color, (x, y), RADIUS)
        label = FONT.render(str(n), True, (0, 0, 0))
        screen.blit(label, (x - label.get_width() // 2, y - label.get_height() // 2))

    # Draw stack/queue panel
    draw_data_structure_panel()

    if algo_running:
        step_text = f"Steps: {algo_index} / {len(algo_state)}"
        step_label = FONT.render(step_text, True, (255, 255, 255))
        pygame.draw.rect(screen, (50, 50, 50), (10, HEIGHT - 40, step_label.get_width() + 20, 30))
        pygame.draw.rect(screen, (100, 100, 100), (10, HEIGHT - 40, step_label.get_width() + 20, 30), 2)
        screen.blit(step_label, (20, HEIGHT - 33))

    pygame.display.flip()
    clock.tick(60)
