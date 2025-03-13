"""Functions to test intersections between primitives.

Note how these functions are written: no allocations, simple return
types, writing values to pre-allocated arrays passed in as
arguments. This all makes the numba conversions easier.
"""

from typing import NamedTuple

import numpy as np
from numba import jit

from .config import PrimitiveKind
from .maths import closest_point_on_cube
from .bodies import RigidBody


class Collision(NamedTuple("Collision", [("normal", np.ndarray),
                                         ("depth", np.ndarray)])):
    """A collision between two rigid bodies."""

    def negate(self) -> "Collision":
        return Collision(-self.normal, self.depth)


@jit(cache=True)
def sphere_test(a_centroid: np.ndarray,
                a_radius: float,
                b_centroid: np.ndarray,
                b_radius: float) -> bool:
    """Test if two spheres intersect."""
    d = b_centroid - a_centroid
    length_2 = np.round(d.dot(d), 3)
    radii = a_radius + b_radius
    radii_2 = np.round(radii * radii, 3)
    return length_2 < radii_2


@jit(cache=True)
def intersect_spheres(a_centroid: np.ndarray,
                      a_radius: float,
                      b_centroid: np.ndarray,
                      b_radius: float,
                      normal: np.ndarray) -> float:
    """Find the intersection between two spheres."""
    d = b_centroid - a_centroid
    distance = np.linalg.norm(d)
    normal[:] = d / distance
    return a_radius + b_radius - distance


FLOAT_MAX = np.finfo(np.float64).max
FLOAT_MIN = np.finfo(np.float64).min


@jit(cache=True)
def intersect_cubes(a_centroid: np.ndarray,
                    a_vertices: np.ndarray,
                    a_normals: np.ndarray,
                    b_centroid: np.ndarray,
                    b_vertices: np.ndarray,
                    b_normals: np.ndarray,
                    normal: np.ndarray) -> float:
    """Find the intersection between two cubes.

    Description:
        This function finds the intersection between two cubes by
        using the separating axis theorem. It returns the depth of
        the intersection and the normal of the collision.
    """
    normals = np.empty((6, 3), np.float64)
    normals[:3] = a_normals
    normals[3:] = b_normals
    depth = FLOAT_MAX
    for i in range(6):
        axis = normals[i]
        a_min = FLOAT_MAX
        b_min = FLOAT_MAX
        a_max = FLOAT_MIN
        b_max = FLOAT_MIN
        for j in range(8):
            a_proj = a_vertices[j].dot(axis)
            b_proj = b_vertices[j].dot(axis)
            a_min = a_proj if a_proj < a_min else a_min
            a_max = a_proj if a_proj > a_max else a_max
            b_min = b_proj if b_proj < b_min else b_min
            b_max = b_proj if b_proj > b_max else b_max

        if a_max <= b_min or b_max <= a_min:
            return -1

        if a_min < b_min:
            if a_max < b_max:
                # a left, b right
                overlap = a_max - b_min
            else:
                # b inside a
                overlap = b_max - a_min
        else:
            if b_max < a_max:
                # b left, a right
                overlap = b_max - a_min
            else:
                # a inside b
                overlap = a_max - b_min

        if overlap < depth:
            depth = overlap
            normal[:] = axis

    dpos = b_centroid - a_centroid
    if dpos.dot(normal) < 0:
        normal *= -1

    return depth


@jit(cache=True)
def intersect_sphere_cube(a_centroid: np.ndarray,
                          a_radius: float,
                          b_centroid: np.ndarray,
                          b_vertices: np.ndarray,
                          b_size: np.ndarray,
                          b_rotation: np.ndarray,
                          b_normals: np.ndarray,
                          normal: np.ndarray) -> float:
    """Find the intersection between a sphere and a cube.

    Description:
        This function finds the intersection between a sphere and a cube
        by using the separating axis theorem. It returns the depth of the
        intersection and the normal of the collision. Note how we need
        to add the normal of the closest point on the cube to the sphere
        centroid (as otherwise it will push spheres away at corners and
        edges).
    """

    normals = np.empty((4, 3), np.float64)
    closest_point_on_cube(a_centroid, b_centroid, b_size, b_rotation, normals[0])
    normals[0] -= a_centroid
    length = np.linalg.norm(normals[0])
    if length < 1e-6:
        normals[:3] = b_normals
        num_normals = 3
    else:
        normals[0] /= np.linalg.norm(normals[0])
        normals[1:] = b_normals
        num_normals = 4

    dpos = (b_centroid - a_centroid)
    depth = FLOAT_MAX
    a_diff = np.empty(3, np.float64)
    for i in range(num_normals):
        axis = normals[i]
        a_diff[:] = a_radius * axis
        a_min = (a_centroid - a_diff).dot(axis)
        a_max = (a_centroid + a_diff).dot(axis)
        if a_min > a_max:
            a_min, a_max = a_max, a_min

        b_min = FLOAT_MAX
        b_max = FLOAT_MIN
        for j in range(8):
            b_proj = b_vertices[j].dot(axis)
            b_min = b_proj if b_proj < b_min else b_min
            b_max = b_proj if b_proj > b_max else b_max

        if a_max <= b_min or b_max <= a_min:
            return -1

        if a_min < b_min:
            if a_max < b_max:
                # a left, b right
                overlap = a_max - b_min
            else:
                # b inside a
                overlap = b_max - a_min
        else:
            if b_max < a_max:
                # b left, a right
                overlap = b_max - a_min
            else:
                # a inside b
                overlap = a_max - b_min

        if overlap < depth:
            depth = overlap
            normal[:] = axis

    if dpos.dot(normal) < 0:
        normal *= -1

    return depth


def collide(a: RigidBody, b: RigidBody) -> Collision:
    if not sphere_test(a.position, a.radius, b.position, b.radius):
        return None

    normal = np.ndarray(3, np.float64)
    match a.kind, b.kind:
        case PrimitiveKind.SPHERE, PrimitiveKind.SPHERE:
            depth = intersect_spheres(a.position, a.radius, b.position, b.radius, normal)
        case PrimitiveKind.SPHERE, _:
            depth = intersect_sphere_cube(a.position, a.radius, b.position, b.transformed_vertices,
                                          b.size, b.rotation, b.transformed_normals, normal)
        case _, PrimitiveKind.SPHERE:
            depth = intersect_sphere_cube(b.position, b.radius, a.position, a.transformed_vertices,
                                          a.size, a.rotation, a.transformed_normals, normal)
            np.negative(normal, out=normal)
        case _:
            depth = intersect_cubes(a.position, a.transformed_vertices, a.transformed_normals,
                                    b.position, b.transformed_vertices, b.transformed_normals,
                                    normal)

    if depth < 0:
        return None

    return Collision(normal, depth)
