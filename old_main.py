import math
import os.path

import geopy.distance

import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
import shutil

import time
from functools import wraps

from shapely import GeometryCollection
from shapely.geometry import Polygon, LineString, GeometryCollection, Point
from shapely.affinity import scale


def timer_decorator(func):
    """
    Decorator to track the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} executed in {execution_time:.4f} seconds.")
        return result
    return wrapper


def get_total_length(geom):
    """
    Recursively compute the total length of the geometry's boundary.
    For a Polygon, we use its exterior; for a LineString, its length;
    for a GeometryCollection, sum over its components.
    """
    if geom.geom_type == 'Polygon':
        return geom.exterior.length
    elif geom.geom_type == 'LineString':
        return geom.length
    elif geom.geom_type == 'GeometryCollection':
        return sum(get_total_length(g) for g in geom.geoms)
    else:
        return 0


def get_node_pos(G, node):
    """Return the (x, y) position of a node in the graph."""
    return G.nodes[node]['pos']


def euclidean_dist(p, q):
    """Compute Euclidean distance between two points p=(x1,y1) and q=(x2,y2)."""
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)


def snap_point_to_nearest_node(G, point):
    """
    Given a point (lon, lat), find the nearest node in a lat/long graph G.
    Uses geopy.distance.geodesic for accurate great-circle distances.
    """
    closest_node = None
    min_dist = float('inf')
    # point is (lon, lat)
    for node in G.nodes:
        node_lon = G.nodes[node]['x']
        node_lat = G.nodes[node]['y']
        # Use geopy; note that geopy expects (lat, lon)
        dist = geopy.distance.geodesic((node_lat, node_lon), (point[1], point[0])).meters
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    return closest_node


def route_length(G, route_nodes):
    """
    Compute the total length of a route (a list of nodes) in the graph G.
    We assume edges have a 'weight' attribute representing distance.
    """
    length = 0.0
    for i in range(len(route_nodes) - 1):
        u, v = route_nodes[i], route_nodes[i+1]
        # Use Dijkstra to get the distance (or you can store it if you have it precomputed)
        length += nx.dijkstra_path_length(G, u, v, weight='weight')
    return length


@timer_decorator
def load_map_as_graph(place: str):
    # # Set up the cache
    # ox.settings.use_cache = True
    # ox.settings.cache_folder = './data/osm_cache'  # specify your cache directory

    # Get the walking network from OSM
    G = ox.graph_from_place(place, network_type='walk')

    return G


@timer_decorator
def save_map_to_disk(G, file_path):
    fig, ax = ox.plot_graph(G, show=False, close=False)
    plt.savefig(file_path)
    plt.close()





def get_rectangle_shape():
    # Define the rectangle's corner coordinates
    # (You can optionally repeat the first point at the end, but Shapely automatically closes polygons.)
    rectangle_coords = [(0, 0), (0, 100), (200, 100), (200, 0)]

    # Create a Polygon from these coordinates
    rectangle_poly = Polygon(rectangle_coords)

    # Wrap it in a GeometryCollection to match the structure used in get_smily_shape()
    rectangle_shape = GeometryCollection([rectangle_poly])

    # Save and return
    save_shape(rectangle_shape, "shape.png")
    return rectangle_shape


def get_smily_shape():
    def create_circle(center=(0, 0), radius=1.0, num_points=100):
        """
        Create a Shapely Polygon approximating a circle.
        """
        cx, cy = center
        angles = np.linspace(0, 2 * math.pi, num_points)
        coords = [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]
        return Polygon(coords)

    def create_arc(center=(0, 0), radius=1.0, start_angle_deg=0, end_angle_deg=180, num_points=50):
        """
        Create a Shapely LineString representing an arc of a circle.
        Angles are in degrees, with 0° on the positive x-axis and 90° on the positive y-axis.
        """
        cx, cy = center
        start_rad = math.radians(start_angle_deg)
        end_rad = math.radians(end_angle_deg)
        angles = np.linspace(start_rad, end_rad, num_points)
        coords = [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]
        return LineString(coords)

    # --- 1. Create the face boundary (a large circle) ---
    face_boundary = create_circle(center=(0, 0), radius=100, num_points=200)

    # --- 2. Create the eyes (two smaller circles) ---
    left_eye = create_circle(center=(-30, 30), radius=10, num_points=50)
    right_eye = create_circle(center=(30, 30), radius=10, num_points=50)

    # --- 3. Create the mouth (an arc) ---
    # This arc starts at 200° and ends at 340°, forming a "smile" near the bottom of the face.
    mouth = create_arc(center=(0, -10), radius=40, start_angle_deg=200, end_angle_deg=340, num_points=50)

    # --- 4. Combine them into a single GeometryCollection ---
    smiley_face = GeometryCollection([face_boundary, left_eye, right_eye, mouth])

    save_shape(smiley_face, "shape.png")
    return smiley_face


def get_heart_shape():
    # Generate parameter t from 0 to 2π
    t = np.linspace(0, 2 * math.pi, 300)

    # Parametric equations for a heart shape:
    # x = 16*sin^3(t)
    # y = 13*cos(t) - 5*cos(2*t) - 2*cos(3*t) - cos(4*t)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

    # Create a list of (x, y) coordinates
    coords = list(zip(x, y))

    # Create a Shapely polygon from the coordinates.
    # Shapely will automatically close the polygon.
    heart_poly = Polygon(coords)

    # Wrap in a GeometryCollection for consistency with the other functions
    heart_shape = GeometryCollection([heart_poly])

    # Optionally, save the shape using your custom function
    save_shape(heart_shape, "shape.png")

    return heart_shape


def scale_shape(shape, target_distance):
    """
    Given a shape (Polygon, LineString, or GeometryCollection) and a target distance,
    scale the shape so that its perimeter approximates the target distance, then sample
    points along its boundary.

    Returns a list of Shapely Point objects representing the sampled route.
    """
    current_length = get_total_length(shape)
    if current_length == 0:
        return []

    # Compute the scale factor to adjust the perimeter
    scale_factor = target_distance / current_length

    # Scale the geometry about its centroid (so it stays in roughly the same location)
    scaled_geom = scale(shape, xfact=scale_factor, yfact=scale_factor, origin='centroid')

    # Sample points along the scaled geometry
    save_shape(scaled_geom, "scaled_shape.png")
    return scaled_geom


def sample_geometry(geom, num_points=100):
    """Return a list of Shapely Point objects sampled along the boundary."""
    # For simplicity, let's only handle a single polygon or linestring here.
    # For multiple shapes, you'd do a bit more logic.
    if geom.geom_type == 'Polygon':
        boundary = geom.exterior
    elif geom.geom_type == 'LineString':
        boundary = geom
    elif geom.geom_type == 'GeometryCollection':
        # Just sample the outer boundary of the first geometry in the collection
        # Or sum them all. This is up to you.
        boundary = geom.geoms[0].exterior if geom.geoms[0].geom_type == 'Polygon' else geom.geoms[0]
    else:
        return []

    L = boundary.length
    points = []
    for i in range(num_points):
        d = (i / (num_points - 1)) * L
        points.append(boundary.interpolate(d))
    return points


def naive_shape_route(G, shape_points):
    """
    Given a list of shape points, snap each point to the nearest node, then
    connect them in order using shortest paths. Return the resulting route (list of nodes).
    """
    snapped_nodes = []
    for p in shape_points:
        snapped_nodes.append(snap_point_to_nearest_node(G, (p.x, p.y)))

    # Build the route by concatenating shortest paths between consecutive snapped nodes
    route = []
    for i in range(len(snapped_nodes) - 1):
        u = snapped_nodes[i]
        v = snapped_nodes[i + 1]
        segment = nx.shortest_path(G, u, v, weight='weight')
        # Avoid duplicating nodes at the junctions
        if i == 0:
            route.extend(segment)
        else:
            route.extend(segment[1:])

    return route


def shape_match_cost(G, route_nodes, shape_points, target_distance):
    """
    A simple cost function that:
      1. Penalizes difference from target_distance.
      2. Penalizes difference in shape by summing Euclidean distances
         between route sample points and shape_points.

    For shape comparison, we'll sample the route in the same number of points as shape_points
    and sum the distances.
    """
    # 2.1: Compute route length
    r_length = route_length(G, route_nodes)
    dist_cost = abs(r_length - target_distance)

    # 2.2: Sample the route in the same number of points as shape_points
    # First, build a LineString for the route using node positions
    route_coords = []
    for i in range(len(route_nodes)):
        route_coords.append(get_node_pos(G, route_nodes[i]))
    # route_coords is a piecewise path, so let's create a single LineString
    route_line = LineString(route_coords)

    # Sample route_line at equal intervals
    shape_count = len(shape_points)
    route_sampled_points = []
    L = route_line.length
    for i in range(shape_count):
        d = (i / (shape_count - 1)) * L  # distance along the route
        route_sampled_points.append(route_line.interpolate(d))

    # Sum of Euclidean distances between each route-sampled point and shape point
    shape_cost = 0.0
    for sp, rp in zip(shape_points, route_sampled_points):
        shape_cost += euclidean_dist((sp.x, sp.y), (rp.x, rp.y))

    # Combine costs with weights
    alpha = 1.0  # weight for distance cost
    beta = 1.0  # weight for shape cost
    total_cost = alpha * dist_cost + beta * shape_cost
    return total_cost


def plot_route_on_graph(G, route_nodes, filename='map_with_route.png'):
    """
    Plot the given route (list of nodes) on top of the OSMnx graph G.
    Saves the result to `filename`.
    """
    # 1. Plot the graph and retrieve figure, axis
    fig, ax = ox.plot_graph(G, show=False, close=False, bgcolor='white', node_color='black', edge_color='gray')

    # 2. Extract the (x, y) positions of each node in the route
    x_coords = []
    y_coords = []
    for node in route_nodes:
        # If you're using OSMnx default attributes, they might be G.nodes[node]['x'] and G.nodes[node]['y']
        # If you manually stored them in G.nodes[node]['pos'] = (x, y), use that instead.
        x, y = G.nodes[node]['pos']  # or (G.nodes[node]['x'], G.nodes[node]['y'])
        x_coords.append(x)
        y_coords.append(y)

    # 3. Plot the route on top of the graph
    ax.plot(x_coords, y_coords, color='red', linewidth=3, alpha=0.8, zorder=5)

    # 4. Save (or show) the figure
    plt.savefig(filename)
    plt.close(fig)


def setup_asset_dir(data_dir_path):
    if os.path.exists(data_dir_path):
        shutil.rmtree(data_dir_path)
    os.makedirs(data_dir_path)


def main(data_dir_path: str, shape, perimeter: float):
    setup_asset_dir(data_dir_path)

    G = load_map_as_graph("Cubbon park, Bangalore")
    save_map_to_disk(G, os.path.join(data_dir_path, "map.png"))

    # shape = get_rectangle_shape()
    # shape = scale_shape(shape, perimeter)
    # shape_points = sample_geometry(shape, num_points=50)
    # route_nodes = naive_shape_route(G, shape_points)
    # route, cost = shape_match_cost(G, route_nodes, shape_points, perimeter)
    # print("Cost", cost)
    # plot_route_on_graph(G, route)


if __name__ == "__main__":
    data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    main(
        data_dir_path=data_dir_path,
        perimeter=5000
    )

