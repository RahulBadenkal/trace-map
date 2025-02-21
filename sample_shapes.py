import math

import numpy as np
from shapely import Point, LineString, Polygon, GeometryCollection
from helpers import normalize_shape


def get_line() -> GeometryCollection:
    shape = GeometryCollection([
        LineString([(0, 0), (2, 0)]),
    ])
    shape = normalize_shape(shape)
    return shape


def get_triangle_shape() -> GeometryCollection:
    shape = GeometryCollection([
        LineString([(0, 0), (0, 2), (1, 1), (0, 0)]),
    ])
    shape = normalize_shape(shape)
    return shape


def get_rectangle_shape() -> GeometryCollection:
    shape = GeometryCollection([
        LineString([(0, 0), (2, 0), (2, 1), (0, 1), (0, 0)]),
        LineString([(0, 0), (2, 0), (2, 1), (0, 1), (0, 0)]),
    ])
    shape = normalize_shape(shape)
    return shape


def get_smily_shape() -> GeometryCollection:
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
    smiley_face = normalize_shape(smiley_face)
    return smiley_face


def get_heart_shape() -> GeometryCollection:
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

    heart_shape = normalize_shape(heart_shape)

    return heart_shape


LINE = get_line()

TRIANGLE = get_triangle_shape()

RECTANGLE = get_rectangle_shape()

SMILEY_FACE = get_smily_shape()

HEART = get_heart_shape()


if __name__ == "__main__":
    from helpers import save_shape

    # shape = Point(1, 1)
    # shape = LineString([(0, 0), (3, 1), (0, 3), (5, 6), (5, 5)])
    # shape = Polygon([(0, 0), (3, 1), (0, 3), (5, 6), (5, 5)])
    shape = RECTANGLE
    save_shape(shape, "./sample.png")

