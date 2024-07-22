import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import heapq

class PathFindingProblem(Problem):
    def __init__(self, start, end, barriers, n_points=5, max_variance=5.0): 
        self.start = start
        self.end = end
        self.barriers = barriers
        self.n_points = n_points
        self.safe_distance = 5.0  # Safe distance threshold
        self.max_variance = max_variance  # Maximum allowed variance in segment lengths
        super().__init__(n_var=2 * n_points,
                         n_obj=3,  # Change to 3 objectives
                         n_constr=2,  # Change to 2 constraints
                         xl=0.0,
                         xu=100.0)
    
    def calculate_penalty(self, point):
        min_distance = float('inf')
        for barrier in self.barriers:
            for vertex in barrier:
                distance = np.linalg.norm(np.array(point) - np.array(vertex))
                if distance < min_distance:
                    min_distance = distance
        if min_distance < self.safe_distance:
            penalty = 1 / (min_distance + 1)  # Inverse penalty within safe distance
        else:
            penalty = 0  # No penalty beyond safe distance
        return penalty
    
    def point_in_polygon(self, point, polygon):
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
    
    def do_lines_intersect(self, A, B, C, D):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def is_collision(self, point1, point2=None):
        for barrier in self.barriers:
            if self.point_in_polygon(point1, barrier):
                return True
            if point2 is not None:
                for i in range(len(barrier)):
                    if self.do_lines_intersect(point1, point2, barrier[i], barrier[(i + 1) % len(barrier)]):
                        return True
        return False   
    
    def smoothness(self, position):
        smooth = 0
        for i in range(1, len(position) - 1):
            alpha = math.atan2(position[i + 1][1] - position[i][1], position[i + 1][0] - position[i][0])
            beta = math.atan2(position[i][1] - position[i - 1][1], position[i][0] - position[i - 1][0])
            angle_diff = abs(alpha - beta)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            if angle_diff > math.pi / 8:  # Penalize sharp turns
                smooth += 10
            smooth += angle_diff
        
        return smooth / len(position)  # Normalize smoothness score

    def uniformity(self, path):
        distances = [np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)]
        variance = np.var(distances)
        return variance

    def _evaluate(self, X, out, *args, **kwargs):
        paths = X.reshape(-1, self.n_points, 2)
        total_distances = []
        penalties = []
        smoothness_scores = []
        collision_constraints = []
        variance_constraints = []

        for path in paths:
            path = np.vstack([self.start, path, self.end])
            total_distance = 0
            total_penalty = 0
            collision = 0

            for i in range(len(path) - 1):
                total_distance += np.linalg.norm(path[i + 1] - path[i])
                total_penalty += self.calculate_penalty(path[i])
                if self.is_collision(path[i], path[i + 1]):
                    collision += 1  # Increment collision count

            smooth = self.smoothness(path)
            variance = self.uniformity(path)
            total_distances.append(total_distance)
            penalties.append(total_penalty)
            smoothness_scores.append(smooth)
            collision_constraints.append(collision)  # Add collision count to constraints
            variance_constraints.append(variance - self.max_variance)  # Add variance constraint

        out["F"] = np.column_stack([total_distances, penalties, smoothness_scores])
        out["G"] = np.column_stack([collision_constraints, variance_constraints])  # Set constraints

def plot_path(barriers, path):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    for barrier in barriers:
        polygon = patches.Polygon(barrier, closed=True, edgecolor='r', facecolor='none')
        ax.add_patch(polygon)

    if path is not None:
        x_coords = path[:, 0]
        y_coords = path[:, 1]
        ax.plot(x_coords, y_coords, 'bo-')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Path Finding with Genetic Algorithm')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def astar(start, end, barriers):
    def heuristic(a, b):
        return np.linalg.norm(a - b)

    def is_valid_point(point):
        if point[0] < 0 or point[0] > 100 or point[1] < 0 or point[1] > 100:
            return False
        for barrier in barriers:
            if point_in_polygon(point, barrier):
                return False
        return True

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

    def a_star_algorithm(start, end):
        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, end), 0, tuple(start), []))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): heuristic(start, end)}

        while open_set:
            _, current_cost, current, path = heapq.heappop(open_set)
            current = np.array(current)

            if np.linalg.norm(current - end) < 1e-2:
                return np.array(path + [current] + [end])

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = current + np.array([dx, dy])
                if is_valid_point(neighbor):
                    tentative_g_score = g_score[tuple(current)] + np.linalg.norm(neighbor - current)
                    if tuple(neighbor) not in g_score or tentative_g_score < g_score[tuple(neighbor)]:
                        came_from[tuple(neighbor)] = tuple(current)
                        g_score[tuple(neighbor)] = tentative_g_score
                        f_score[tuple(neighbor)] = tentative_g_score + heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score[tuple(neighbor)], tentative_g_score, tuple(neighbor), path + [current]))

        return []

    path = a_star_algorithm(start, end)
    return np.array(path)

# Define barriers, start, end points
barriers = [
    [(10, 10), (20, 10), (15, 20)],
    [(30, 30), (50, 30), (50, 50), (30, 50)],
    [(70, 70), (80, 65), (85, 75), (75, 80)]
]

start = np.array([0, 0])
end = np.array([90, 90])

# Compute A* path
astar_path = astar(start, end, barriers)
astar_path = np.vstack([start, astar_path, end])

plot_path(barriers, astar_path)

# Generate initial population
n_individuals = 10
initial_population = np.tile(astar_path.flatten(), (n_individuals, 1))

# Define the problem
problem = PathFindingProblem(start, end, barriers, n_points=5)

# Define the algorithm
algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)

# Perform the optimization
res = minimize(problem,
               algorithm,
               termination=('n_gen', 200),
               seed=1,
               verbose=True,
               X=initial_population)  # Pass the initial population

# Get the optimal path
optimal_path = res.X.reshape(-1, problem.n_points, 2)[0]
optimal_path = np.vstack([start, optimal_path, end])

# Plot the environment and the path found
plot_path(barriers, optimal_path)
