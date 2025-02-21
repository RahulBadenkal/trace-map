import math
import os.path
from typing import Dict, List, Optional, Tuple

import geopy.distance

import networkx as nx
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import shutil

from shapely import GeometryCollection, LineString

from helpers import timer_decorator, save_shape, scale_shape_to_target_length, get_total_length, extract_coords, \
    discrete_frechet_distance, offset_to_origin


def setup_asset_dir(data_dir_path):
    if os.path.exists(data_dir_path):
        shutil.rmtree(data_dir_path)
    os.makedirs(data_dir_path)


def load_map_as_graph(place: str):
    # # Set up the cache
    # ox.settings.use_cache = True
    # ox.settings.cache_folder = './data/osm_cache'  # specify your cache directory

    # Get the walking network from OSM
    G = ox.graph_from_place(place, network_type='walk')

    return G


def save_map_to_disk(G, starting_node, file_path: str):
    # Plot the graph without closing it
    fig, ax = ox.plot_graph(G, show=False, close=False, node_size=10, edge_linewidth=0.5)

    # Get the coordinates of the start node
    start_x, start_y = G.nodes[starting_node]['x'], G.nodes[starting_node]['y']

    # Highlight the starting node in red
    ax.scatter(start_x, start_y, color='red', s=15, zorder=3, label="Start Node")

    # Add legend
    ax.legend()

    # Save the plot to a file
    plt.savefig(file_path, dpi=300)
    plt.close()


def densify_geometry(geom, max_segment_length=1.0):
    """
    Given a LineString (or similar linear geometry), insert additional vertices so that
    no segment is longer than max_segment_length.
    """
    if geom.is_empty or geom.length == 0:
        return geom

    total_length = geom.length
    num_segments = int(np.ceil(total_length / max_segment_length))
    points = [geom.interpolate(float(i) / num_segments, normalized=True) for i in range(num_segments + 1)]
    return LineString([pt.coords[0] for pt in points])


def candidate_path_to_geometry_collection(G, path, max_segment_length=1.0):
    """
    Convert a candidate path (list of node IDs) into a GeometryCollection.
    For each consecutive pair of nodes, if the edge has a 'geometry' attribute,
    that curved geometry is densified; otherwise, a straight line is created and densified.
    """
    edge_geometries = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if 'geometry' in G[u][v]:
            edge_geom = G[u][v]['geometry']
        else:
            p1 = (G.nodes[u]['x'], G.nodes[u]['y'])
            p2 = (G.nodes[v]['x'], G.nodes[v]['y'])
            edge_geom = LineString([p1, p2])
        densified_geom = densify_geometry(edge_geom, max_segment_length)
        edge_geometries.append(densified_geom)
    geom = GeometryCollection(edge_geometries)
    return geom


def get_all_paths(G, starting_node, track_length: float) -> Tuple[List[List[str]], List[GeometryCollection]]:
    total_paths = {"count": 0}
    all_paths = []

    def dfs(node, current_path, current_length, visited):
        # Found a valid cycle (must contain at least one edge besides the start).
        if node == starting_node and current_length >= track_length and len(current_path) > 1:
            all_paths.append(current_path.copy())
            print("Found a valid path:", len(all_paths))
            return

        # If the length exceeds track_length and we haven't closed the cycle, prune.
        if current_length >= track_length:
            total_paths["count"] += 1
            print(total_paths["count"], current_length)
            return

        for neighbor in G.neighbors(node):
            if visited.get(neighbor, 0) >= 2:
                continue

            edge_data = G.get_edge_data(node, neighbor)
            # For each parallel edge between node and neighbor.
            for key in edge_data:
                distance = edge_data[key].get('length', 0)
                new_length = current_length + distance

                # Add neighbor to the path and mark it visited.
                current_path.append(neighbor)
                visited[neighbor] = visited.get(neighbor, 0) + 1

                dfs(neighbor, current_path, new_length, visited)

                # Backtrack: remove neighbor and decrement its visit count.
                visited[neighbor] -= 1
                current_path.pop()

    # Start DFS with the starting_node already in the path.
    dfs(starting_node, [starting_node], 0, {starting_node: 1})

    # Convert each candidate node path to a GeometryCollection using your densification logic.
    candidate_geometries = []
    for path in all_paths:
        geom_collection = candidate_path_to_geometry_collection(G, path, 5)
        candidate_geometries.append(geom_collection)

    return all_paths, candidate_geometries

def shape_discrete_distance(shape1: GeometryCollection, shape2: GeometryCollection,
                            translation_invariant: bool = False) -> float:
    """
    Compute the discrete Fréchet distance between two shapes represented as Shapely GeometryCollections.

    Parameters:
      shape1, shape2: Shapely GeometryCollection objects (or any geometry type supported by extract_coords).
      translation_invariant: If True, center both sets of coordinates (by subtracting their centroids)
                             before computing the distance.

    Returns:
      The discrete Fréchet distance between the two shapes.
    """
    coords1 = extract_coords(shape1)
    coords2 = extract_coords(shape2)

    # Use Shapely's centroid attribute to center coordinates.
    centroid1 = shape1.centroid
    centroid2 = shape2.centroid
    coords1 = [(x - centroid1.x, y - centroid1.y) for x, y in coords1]
    coords2 = [(x - centroid2.x, y - centroid2.y) for x, y in coords2]

    return discrete_frechet_distance(coords1, coords2)


def find_best_match(G, shape: GeometryCollection, paths: List[GeometryCollection]) -> int:
    if len(paths) == 0:
        return -1
    if len(paths) == 1:
        return 0

    best_path_index = -1
    best_distance = float('inf')
    for index, path in enumerate(paths):
        distance = shape_discrete_distance(path, shape)
        # Debug print to see each candidate's distance.
        # print(f"Path {path} has Fréchet distance: {distance:.3f}")

        if distance < best_distance:
            best_distance = distance
            best_path_index = index

    return best_path_index


def draw_path_on_graph(G, path, geom, file_path: str, show_direction=False):
    """
    Plot the graph with the candidate path highlighted in red and add arrows along the route.

    Parameters:
      G: The OSMnx graph.
      path: A list of node IDs representing the candidate path.
      file_path: Path to save the resulting image.
    """
    # Plot the graph with the candidate path highlighted.
    fig, ax = ox.plot_graph_route(
        G,
        route=path,
        route_color='r',
        route_linewidth=3,
        node_size=0,
        show=False,
        close=False,
    )

    if show_direction:
        # We'll place a short arrow near the midpoint of each edge.
        # 'arrow_len' is how long the arrow line will be (in the same units as the graph).
        # Setting lw=0 hides the arrow's connecting line, leaving mostly the arrowhead.
        arrow_len = 1.0  # Adjust based on your graph's scale (e.g., meters).
        arrowprops = dict(
            arrowstyle='-|>',  # Arrow with a triangular head, no tail.
            color='blue',
            lw=0,  # No visible line, just the head.
            mutation_scale=12,  # Scale the arrowhead size.
        )

        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            x_start, y_start = G.nodes[start_node]['x'], G.nodes[start_node]['y']
            x_end, y_end = G.nodes[end_node]['x'], G.nodes[end_node]['y']

            # Create a line between these two nodes to measure distance, midpoint, etc.
            segment_line = LineString([(x_start, y_start), (x_end, y_end)])
            seg_len = segment_line.length
            mid_dist = seg_len / 2

            # Define the start/end of the short arrow around the midpoint of the segment.
            half_arrow = arrow_len / 2
            if seg_len < arrow_len:
                # If the segment is shorter than arrow_len, just use the entire segment.
                start_pt = segment_line.interpolate(0)
                end_pt = segment_line.interpolate(seg_len)
            else:
                # Otherwise, place a short arrow in the middle.
                start_pt = segment_line.interpolate(mid_dist - half_arrow)
                end_pt = segment_line.interpolate(mid_dist + half_arrow)

            # Draw the arrow. Since lw=0, you'll mostly see the arrowhead.
            ax.annotate(
                "",
                xy=(end_pt.x, end_pt.y),
                xytext=(start_pt.x, start_pt.y),
                arrowprops=arrowprops
            )

    # 3) Display the total route distance as a label in the upper-left corner.
    total_dist = get_total_length(geom)
    dist_label = f"Distance: {total_dist:.1f} m"  # Assuming 'length' is in meters
    ax.text(
        0.02, 0.98,
        dist_label,
        transform=ax.transAxes,
        color='white',
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.5)
    )

    # Save the figure to disk.
    fig.savefig(file_path, dpi=300)
    plt.close(fig)  # Close the figure to free up memory.


def main(data_dir_path: str, place: str, starting_position: Dict, shape: GeometryCollection, track_length: float):
    setup_asset_dir(data_dir_path)

    G = load_map_as_graph(place=place)
    starting_node = ox.distance.nearest_nodes(G, starting_position["long"], starting_position["lat"])
    save_map_to_disk(G, starting_node, os.path.join(data_dir_path, "map.png"))

    shape = scale_shape_to_target_length(shape, track_length)
    save_shape(shape=shape, file_path=os.path.join(data_dir_path, "shape.png"))

    print("Started finding all paths")
    all_routes, all_route_geoms = get_all_paths(G=G, starting_node=starting_node, track_length=track_length)
    print(f"Total routes found: {len(all_routes)}")

    best_match_index = find_best_match(G, shape, all_route_geoms)
    if best_match_index == -1:
        print("No match found")
        return
    best_path = all_routes[best_match_index]
    draw_path_on_graph(G, best_path, all_route_geoms[best_match_index], os.path.join(data_dir_path, "best_route.png"))
    print(f"Best path index: {best_match_index}")

    # Plot the path on graph
    # shape_points = sample_geometry(shape, num_points=50)
    # route_nodes = naive_shape_route(G, shape_points)
    # route, cost = shape_match_cost(G, route_nodes, shape_points, perimeter)
    # print("Cost", cost)
    # plot_route_on_graph(G, route)


if __name__ == "__main__":
    from sample_shapes import LINE, TRIANGLE, RECTANGLE, SMILEY_FACE, HEART

    data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    latitude = 12 + 58 / 60 + 35.9 / 3600  # 12°58'35.9"N -> 12.9766389
    longitude = 77 + 35 / 60 + 28.2 / 3600  # 77°35'28.2"E -> 77.5911667

    latitude = 12 + 58 / 60 + 34.3 / 3600
    longitude = 77 + 35 / 60 + 29.0 / 3600

    main(
        data_dir_path=data_dir_path,
        place="Cubbon park, Bangalore",
        starting_position={"lat": latitude, "long": longitude},
        shape=TRIANGLE,
        track_length=300
    )

