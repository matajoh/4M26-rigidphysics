"""Module providing routines for collision detection."""

import math
from typing import NamedTuple

from pygame import Vector2

from .bodies import Circle, Polygon, RigidBody


class Collision(NamedTuple("Collision", [("normal", Vector2), ("depth", float)])):
    def reverse(self) -> "Collision":
        return Collision(-self.normal, self.depth)


def intersect_circle_circle(a: Circle, b: Circle) -> Collision:
    """Determine if two circles intersect and return the collision."""
    diff = b.position - a.position
    diff_length2 = diff.dot(diff)
    # due to the graphics system, a circle is visually a bit smaller
    # than its radius, so this is a bit of a hack to correct for that
    radius_sum = (a.radius + b.radius) * 0.97
    if diff_length2 >= radius_sum**2:
        # this test is equivalent to a test of the actual length but much
        # cheaper to compute by avoiding the square root
        return None

    # this way we only compute the square root if we have to
    diff_length = math.sqrt(diff_length2)
    normal = diff / diff_length
    depth = radius_sum - diff_length
    return Collision(normal, depth)


def closest_vertex_on_polygon(point: Vector2, poly: Polygon) -> Vector2:
    """Find the closest vertex on a polygon to a point."""
    closest = None
    dist = float("inf")
    for v in poly.transformed_vertices:
        d = (point - v).length_squared()
        if d < dist:
            closest = v
            dist = d

    return closest


Projection = NamedTuple("Projection", [("min", float), ("max", float)])


def project_polygon_onto_axis(poly: Polygon, axis: Vector2) -> Projection:
    """Project a polygon onto an axis and return the projection."""
    min_proj = float("inf")
    max_proj = float("-inf")
    for v in poly.transformed_vertices:
        proj = v.dot(axis)
        min_proj = min(min_proj, proj)
        max_proj = max(max_proj, proj)

    return Projection(min_proj, max_proj)


def project_circle_onto_axis(circle: Circle, axis: Vector2) -> Projection:
    """Project a circle onto an axis and return the projection."""
    center = circle.position.dot(axis)
    # due to the graphics system, a circle is visually a bit smaller
    # than its radius, so this is a bit of a hack to correct for that
    return Projection(center - circle.radius * 0.97, center + circle.radius * 0.97)


def intersect_circle_polygon(circle: Circle, poly: Polygon) -> Collision:
    """Determine if a circle and a polygon intersect and return the collision."""

    # we need to add this extra normal to account for when the circle is
    # at a corner.
    closest_point = closest_vertex_on_polygon(circle.position, poly)
    diff = closest_point - circle.position
    normals = poly.transformed_normals + [diff.normalize()]

    axis = None
    min_depth = float("inf")
    for normal in normals:
        circle_proj = project_circle_onto_axis(circle, normal)
        rect_proj = project_polygon_onto_axis(poly, normal)
        if circle_proj.max < rect_proj.min or rect_proj.max < circle_proj.min:
            return None

        depth = min(circle_proj.max - rect_proj.min, rect_proj.max - circle_proj.min)
        if depth < min_depth:
            min_depth = depth
            axis = normal

    dpos = poly.position - circle.position
    if dpos.dot(axis) < 0:
        axis = -axis

    return Collision(axis, min_depth)


def intersect_polygon_polygon(a: Polygon, b: Polygon) -> Collision:
    """Determine if two polygons intersect and return the collision."""

    normals = a.transformed_normals + b.transformed_normals
    axis = None
    min_depth = float("inf")
    for normal in normals:
        a_proj = project_polygon_onto_axis(a, normal)
        b_proj = project_polygon_onto_axis(b, normal)
        if a_proj.max < b_proj.min or b_proj.max < a_proj.min:
            return None

        depth = min(a_proj.max - b_proj.min, b_proj.max - a_proj.min)
        if depth < min_depth:
            min_depth = depth
            axis = normal

    dpos = b.position - a.position
    if dpos.dot(axis) < 0:
        axis = -axis

    return Collision(axis, min_depth)


def detect_collision(a: RigidBody, b: RigidBody) -> Collision:
    if isinstance(a, Circle):
        if isinstance(b, Circle):
            return intersect_circle_circle(a, b)

        return intersect_circle_polygon(a, b)
    elif isinstance(a, Polygon):
        if isinstance(b, Circle):
            collision = intersect_circle_polygon(b, a)
            return collision.reverse() if collision else None

        return intersect_polygon_polygon(a, b)
