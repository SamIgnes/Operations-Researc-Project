import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def heuristic(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def calculate_penalty(point, barriers):
    min_distance = float('inf')
    for barrier in barriers:
        for vertex in barrier:
            distance = heuristic(point, vertex)
            if distance < min_distance:
                min_distance = distance
    return 10 / (min_distance + 1)  # Penalizzazione inversa


def EdgeCost(node, neighbor, barriers):
    edgecost = heuristic(node, neighbor) + calculate_penalty(neighbor, barriers) 
    return edgecost

def create_graph(barriers, width, height):
    G = nx.Graph()
    step_size = 1  # Step size for moving to neighbors

    # Add nodes
    for x in range(0, width, step_size):
        for y in range(0, height, step_size):
            if not is_collision((x, y), barriers):
                G.add_node((x, y))

    # Add edges
    for node in G.nodes():
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (node[0] + dx, node[1] + dy)
                if neighbor in G.nodes():
                    G.add_edge(node, neighbor, weight= EdgeCost(node, neighbor, barriers))

    return G

def is_collision(point, barriers):
    for barrier in barriers:
        if point_in_polygon(point, barrier):
            return True
    return False

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def plot_environment_and_path(barriers, paths):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    for barrier in barriers:
        polygon = patches.Polygon(barrier, closed=True, edgecolor='r', facecolor='none')
        ax.add_patch(polygon)

    colors = ['b', 'g', 'm', 'c', 'y']  # Esempio di colori diversi per i percorsi

    for idx, path in enumerate(paths):
        if path:
            x_coords = [point[0] for point in path]
            y_coords = [point[1] for point in path]
            ax.plot(x_coords, y_coords, linestyle='-', color=colors[idx % len(colors)])

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Path Finding')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

def generate_random_polygon(max_sides, min_radius, max_radius):
    num_sides = random.randint(3, max_sides)
    angle_step = 360 / num_sides
    points = []
    for i in range(num_sides):
        angle = math.radians(i * angle_step)
        radius = random.uniform(min_radius, max_radius)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points.append((x, y))
    return points

def is_intersecting(polygon, barriers):
    poly_patch = patches.Polygon(polygon, closed=True)
    for barrier in barriers:
        barrier_patch = patches.Polygon(barrier, closed=True)
        if poly_patch.get_path().intersects_path(barrier_patch.get_path()):
            return True
    return False

def create_non_intersecting_polygons(n, max_sides=6, min_radius=10, max_radius=20, width=100, height=100, start=(0, 0), end=(100, 100)):
    barriers = []
    attempts = 0
    while len(barriers) < n and attempts < 1000:
        attempts += 1
        polygon = generate_random_polygon(max_sides, min_radius, max_radius)
        centroid_x = random.uniform(0, width)
        centroid_y = random.uniform(0, height)
        translated_polygon = [(x + centroid_x, y + centroid_y) for x, y in polygon]
        
        # Check if start and end points are inside the translated polygon
        if not point_in_polygon(start, translated_polygon) and not point_in_polygon(end, translated_polygon):
            if not is_intersecting(translated_polygon, barriers):
                barriers.append(translated_polygon)
    
    return barriers

def plot_polygons(barriers):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    for barrier in barriers:
        polygon = patches.Polygon(barrier, closed=True, edgecolor='r', facecolor='none')
        ax.add_patch(polygon)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Generated Non-Intersecting Polygons')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_cost = float('inf')

def smoothness(position):
    smooth = 0
    for i in range(1,len(position)-2):

        if (position[i+1][0]-position[i][0] == 0) & (position[i][0]-position[i-1][0] == 0) :
            smooth = math.pi
        elif position[i+1][0]-position[i][0] == 0 :
            smooth = abs(math.pi+math.pi/2-math.atan((position[i][1]-position[i-1][1])/(position[i][0]-position[i-1][0])))

        elif position[i][0]-position[i-1][0] == 0 :
            smooth = abs(math.pi+math.pi/2-math.atan((position[i+1][1]-position[i][1])/(position[i+1][0]-position[i][0])))

        else:
            smooth += abs(math.pi+math.atan((position[i+1][1]-position[i][1])/(position[i+1][0]-position[i][0]))-math.atan((position[i][1]-position[i-1][1])/(position[i][0]-position[i-1][0])))

    return smooth

def evaluate_cost(position, end, barriers):
    total_cost = 0
    #for i in range(len(position) - 1):
    #    total_cost += heuristic(position[i], position[i+1])
    
    #total_cost += heuristic(position[-1], end)
    #total_cost += calculate_penalty(position[-1], barriers)
    total_cost += smoothness(position)

    return total_cost

def pso_optimization(start, end, barriers, best_path_astar):
    num_particles = 100
    max_iterations = 200
    inertia_weight = 0.7
    cognitive_param = 1.5
    social_param = 1.5

    particles = []
    for _ in range(num_particles):
        # Initialize particles around the best path found by A*
        initial_position = [best_path_astar[0]] + \
                   [(point[0] + random.uniform(-1, +1), point[1] + random.uniform(-1, +1)) 
                    for point in best_path_astar[1:-1]] + \
                   [best_path_astar[-1]]
        initial_velocity = [(0, 0) for _ in initial_position]
        particles.append(Particle(initial_position, initial_velocity))

    global_best_position = best_path_astar

    global_best_cost = evaluate_cost(global_best_position, end, barriers)

    for _ in range(max_iterations):
        for particle in particles:
            # Update particle velocity
            for i in range(len(particle.position)):
                r1, r2 = random.uniform(0, 1), random.uniform(0, 1)
                cognitive_velocity = (cognitive_param * r1 * 
                                      (particle.best_position[i][0] - particle.position[i][0]), 
                                      cognitive_param * r1 * 
                                      (particle.best_position[i][1] - particle.position[i][1]))
                social_velocity = (social_param * r2 * 
                                   (global_best_position[i][0] - particle.position[i][0]), 
                                   social_param * r2 * 
                                   (global_best_position[i][1] - particle.position[i][1]))
                particle.velocity[i] = (inertia_weight * particle.velocity[i][0] + cognitive_velocity[0] + social_velocity[0],
                                        inertia_weight * particle.velocity[i][1] + cognitive_velocity[1] + social_velocity[1])
            
            # Update particle position
            new_position = [(particle.position[i][0] + particle.velocity[i][0], 
                             particle.position[i][1] + particle.velocity[i][1]) 
                            for i in range(len(particle.position))]
            
            # Evaluate new position
            new_cost = evaluate_cost(new_position, end, barriers)

            # Update particle's best position if improved
            if new_cost < particle.best_cost:
                particle.best_position = new_position
                particle.best_cost = new_cost
            
            # Update global best if improved
            if new_cost < global_best_cost:
                global_best_position = new_position
                global_best_cost = new_cost

    return global_best_position

width = 100
height = 100
# Generate 10 non-intersecting polygons excluding start and end points
start = (random.randint(0, width), random.randint(0, height))
end = (random.randint(0, width), random.randint(0, height))
barriers = create_non_intersecting_polygons(10, start=start, end=end)

# Create graph
graph = create_graph(barriers, width, height)

# Find the path using A* algorithm
best_path_astar = nx.astar_path(graph, start, end, heuristic=heuristic)

plot_environment_and_path(barriers, [best_path_astar])

# Ottimizza il percorso trovato con PSO
optimized_path = pso_optimization(start, end, barriers, best_path_astar)

# Plot percorso ottimizzato con PSO
plot_environment_and_path(barriers, [optimized_path])

plt.show()