import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def generate_random_polygon(max_sides, min_radius, max_radius, seed=None):
    if seed is not None:
        random.seed(seed)
    
    num_sides = random.randint(3, max_sides)
    angle_step = 360 / num_sides
    points = []
    for i in range(num_sides):
        angle = np.radians(i * angle_step)
        radius = random.uniform(min_radius, max_radius)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append((x, y))
    return points

def is_intersecting(polygon, barriers):
    poly_patch = patches.Polygon(polygon, closed=True)
    for barrier in barriers:
        barrier_patch = patches.Polygon(barrier, closed=True)
        if poly_patch.get_path().intersects_path(barrier_patch.get_path()):
            return True
    return False

def create_non_intersecting_polygons(n, max_sides=6, min_radius=10, max_radius=20, width=100, height=100, start=(0, 0), end=(90, 90), seed=None):
    if seed is not None:
        seed = seed*1000
        random.seed(seed)
    
    barriers = []
    attempts = 0
    while len(barriers) < n and attempts < 1000:
        attempts += 1
        polygon = generate_random_polygon(max_sides, min_radius, max_radius, seed+attempts)
        centroid_x = random.uniform(0, width)
        centroid_y = random.uniform(0, height)
        translated_polygon = [(x + centroid_x, y + centroid_y) for x, y in polygon]
        
        # Check if start and end points are inside the translated polygon
        if not point_in_polygon(start, translated_polygon) and not point_in_polygon(end, translated_polygon):
            if not is_intersecting(translated_polygon, barriers):
                barriers.append(translated_polygon)
    
    return barriers

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
