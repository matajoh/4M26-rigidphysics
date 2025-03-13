"""Module providing routines for contact point localisation."""

from typing import Tuple
from pygame import Vector2

from .bodies import Circle, Polygon, RigidBody
from .collisions import Collision


def point_segment_distance(p: Vector2, a: Vector2, b: Vector2) -> float:
    """Find the distance between a point and a line segment."""
    ab = b - a
    ap = p - a
    proj = ap.dot(ab)
    ab_length2 = ab.dot(ab)
    d = proj / ab_length2
    if d <= 0:
        closest = a
    elif d >= 1:
        closest = b
    else:
        closest = a + ab * d

    return (p - closest).length_squared()


def are_different(a: Vector2, b: Vector2, tol: float) -> bool:
    return abs(a.x - b.x) > tol or abs(a.y - b.y) > tol


def find_contact_points_polygon_polygon(a: Polygon, b: Polygon, tol=1e-5) -> Tuple[Vector2, None]:
    """Find the contact points between two polygons."""
    num_a_vertices = len(a.vertices)
    min_dist = float("inf")
    closest0 = None
    closest1 = None
    for i in range(num_a_vertices):
        v0 = a.transformed_vertices[i]
        v1 = a.transformed_vertices[(i + 1) % num_a_vertices]
        for p in b.transformed_vertices:
            d = point_segment_distance(p, v0, v1)
            if d < min_dist - tol:
                min_dist = d
                closest0 = p
                closest1 = None
            elif d < min_dist + tol and are_different(p, closest0, tol):
                closest1 = p

    num_b_vertices = len(b.vertices)
    for i in range(num_b_vertices):
        v0 = b.transformed_vertices[i]
        v1 = b.transformed_vertices[(i + 1) % num_b_vertices]
        for p in a.transformed_vertices:
            d = point_segment_distance(p, v0, v1)
            if d < min_dist - tol:
                min_dist = d
                closest0 = p
                closest1 = None
            elif d < min_dist + tol and are_different(p, closest0, tol):
                closest1 = p

    return closest0, closest1


def separate(a: RigidBody, b: RigidBody, collision: Collision):
    if not a.physics:
        # b does not move
        b.position += collision.normal * collision.depth
    elif not b.physics:
        # a does not move
        a.position -= collision.normal * collision.depth
    else:
        a.position -= collision.normal * collision.depth * 0.5
        b.position += collision.normal * collision.depth * 0.5


def find_contact_points(a: RigidBody,
                        b: RigidBody,
                        collision: Collision) -> Tuple[Vector2, Vector2]:
    separate(a, b, collision)
    if isinstance(a, Circle):
        return a.position + collision.normal * a.radius, None
    elif isinstance(a, Polygon):
        if isinstance(b, Circle):
            return b.position - collision.normal * b.radius, None

        return find_contact_points_polygon_polygon(a, b)
