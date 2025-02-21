import math
import time
from functools import wraps
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from shapely.geometry import GeometryCollection, Point, LineString, Polygon
from shapely.affinity import scale, translate


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
        print(f"{func.__name__} -> {execution_time:.4f} seconds.")
        return result
    return wrapper


def save_shape(shape: GeometryCollection, file_path: str, name="Shape"):
    def plot_shapely_geom(ax, geom, **kwargs):
        if geom.geom_type == 'Point':
            ax.scatter(geom.x, geom.y, **kwargs)
        elif geom.geom_type == 'MultiPoint':
            x, y = zip(*[(p.x, p.y) for p in geom.geoms])
            ax.scatter(x, y, **kwargs)
        elif geom.geom_type == 'LineString':
            x, y = geom.xy
            ax.plot(x, y, **kwargs)
        elif geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax.plot(x, y, **kwargs)
        elif geom.geom_type == 'GeometryCollection':
            for g in geom.geoms:
                plot_shapely_geom(ax, g, **kwargs)
        else:
            # Handle other geometry types (Point, MultiPolygon, etc.) if needed
            pass

    # Visualize the rectangle and smiley face
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_shapely_geom(ax, shape, color='black')
    ax.set_aspect('equal', 'box')
    ax.set_title(name)
    ax.grid(True)

    # Save the plot to a file
    plt.savefig(file_path)
    plt.close()


def get_total_length(shape: GeometryCollection) -> float:
    """
    Recursively compute the total length of the geometry's boundary.
    For a Polygon, we use its exterior; for a LineString, its length;
    for a GeometryCollection, sum over its components.
    """
    if shape.geom_type == 'Polygon':
        return shape.exterior.length
    elif shape.geom_type == 'LineString':
        return shape.length
    elif shape.geom_type == 'GeometryCollection':
        return sum(get_total_length(g) for g in shape.geoms)
    else:
        return 0


def offset_to_origin(shape: GeometryCollection) -> GeometryCollection:
    min_x, min_y, max_x, max_y = shape.bounds
    translated_geom = translate(shape, xoff=-min_x, yoff=-min_y)
    return translated_geom


def normalize_shape(shape: GeometryCollection):
    """
    Scales/transforms the shape so that it fits in a 1x1 box starting from (0, 0)
    """
    # Get bounding box of the entire collection
    min_x, min_y, max_x, max_y = shape.bounds

    # Compute width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y
    scale_value = max(width, height)

    # Normalize the geometry
    scaled_geom = scale(shape, xfact=1 / scale_value, yfact=1 / scale_value, origin="centroid")
    translated_geom = offset_to_origin(scaled_geom)

    return translated_geom


def scale_shape_to_target_length(shape: GeometryCollection, length: float) -> GeometryCollection:
    """
    Given a shape (Polygon, LineString, or GeometryCollection) and a target distance,
    scale the shape so that its perimeter approximates the target distance, then sample
    points along its boundary.

    Returns a list of Shapely Point objects representing the sampled route.
    """
    current_length = get_total_length(shape)

    # Compute the scale factor to adjust the perimeter
    scale_factor = length / current_length

    # Scale the geometry about its centroid (so it stays in roughly the same location)
    scaled_geom = scale(shape, xfact=scale_factor, yfact=scale_factor, origin='centroid')
    translated_geom = offset_to_origin(scaled_geom)

    return translated_geom


def extract_coords(geom: GeometryCollection):
    """
    Extract a list of (x, y) coordinates from a Shapely geometry.
    If the geometry is a GeometryCollection with more than one element,
    we combine them using a unary union.
    """
    if geom.is_empty:
        return []

    if geom.geom_type == 'LineString':
        return list(geom.coords)
    elif geom.geom_type == 'MultiLineString':
        coords = []
        for line in geom.geoms:
            coords.extend(list(line.coords))
        return coords
    elif geom.geom_type == 'Polygon':
        # Use the exterior ring for a polygon.
        return list(geom.exterior.coords)
    elif geom.geom_type == 'GeometryCollection':
        # If there's only one geometry, extract from it.
        if len(geom.geoms) == 1:
            return extract_coords(geom.geoms[0])
        else:
            # Combine geometries into one.
            union_geom = unary_union(geom.geoms)
            return extract_coords(union_geom)
    else:
        # Attempt to use the coords attribute if available.
        if hasattr(geom, 'coords'):
            return list(geom.coords)
        else:
            raise ValueError("Unsupported geometry type: " + geom.geom_type)


def discrete_frechet_distance(P, Q):
    """
    Compute the discrete Fréchet distance between two curves P and Q.
    P and Q are lists of (x, y) points.
    """
    n = len(P)
    m = len(Q)
    # Create a 2D cache matrix initialized with -1.
    ca = [[-1 for _ in range(m)] for _ in range(n)]

    def c(i, j):
        if ca[i][j] > -1:
            return ca[i][j]
        dist = math.dist(P[i], Q[j])
        if i == 0 and j == 0:
            ca[i][j] = dist
        elif i == 0 and j > 0:
            ca[i][j] = max(c(0, j - 1), dist)
        elif j == 0 and i > 0:
            ca[i][j] = max(c(i - 1, 0), dist)
        elif i > 0 and j > 0:
            ca[i][j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), dist)
        else:
            ca[i][j] = float('inf')
        return ca[i][j]

    return c(n - 1, m - 1)