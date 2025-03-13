"""Module providing functions for finding contact points between rigid bodies."""

from typing import Tuple
import numpy as np
from numba import jit


from .collisions import Collision
from .config import PrimitiveKind
from .geometry import CUBE_EDGE_INDEX
from .maths import (
    closest_points,
    find_edges,
    insert,
    squared_distance_to_cube,
)
from .bodies import RigidBody


INV_TOL = 64
TOL = 1.0 / INV_TOL
TOL_2 = TOL * TOL


@jit(cache=True)
def find_contact_point_cube(p: np.ndarray,
                            translation: np.ndarray,
                            size: np.ndarray,
                            rotation: np.ndarray,
                            contacts: np.ndarray,
                            min_dist: float,
                            num_contacts: int) -> Tuple[int, float]:
    """Find the contact point between a point and a cube."""
    dist2 = squared_distance_to_cube(p, translation, size, rotation)

    if dist2 > min_dist + TOL_2:
        return num_contacts, min_dist

    contact = (p * INV_TOL).astype(np.int64)
    if dist2 < min_dist - TOL_2:
        contacts[0] = contact
        return 1, dist2

    num_contacts = insert(contacts, contact, num_contacts)
    return num_contacts, min_dist


@jit(cache=True)
def find_contacts_edges(a_vertices: np.ndarray,
                        num_a_edges: int,
                        a_edges: np.ndarray,
                        b_vertices: np.ndarray,
                        num_b_edges: int,
                        b_edges: np.ndarray,
                        contacts: np.ndarray,
                        min_dist: float,
                        num_contacts: int) -> Tuple[int, float]:
    """Find the contact points between two sets of edges."""
    edges = np.empty((2, 2, 3), np.float64)
    points = np.empty((2, 3), np.float64)
    for i in range(num_a_edges):
        edges[0] = a_vertices[a_edges[i]]
        for j in range(num_b_edges):
            edges[1] = b_vertices[b_edges[j]]
            if closest_points(edges, points) == 0:
                continue

            diff = points[0] - points[1]
            dist = diff.dot(diff)

            if dist > min_dist + TOL_2:
                continue

            contact = ((points[0] + points[1]) * 0.5 * INV_TOL).astype(np.int64)
            if dist < min_dist - TOL_2:
                contacts[0] = contact
                num_contacts = 1
                min_dist = dist
            else:
                num_contacts = insert(contacts, contact, num_contacts)

    return num_contacts, min_dist


@jit(cache=True)
def find_contact_points_cube_cube(a_centroid: np.ndarray,
                                  a_vertices: np.ndarray,
                                  a_size: np.ndarray,
                                  a_rotation: np.ndarray,
                                  b_centroid: np.ndarray,
                                  b_vertices: np.ndarray,
                                  b_size: np.ndarray,
                                  b_rotation: np.ndarray,
                                  normal: np.ndarray,
                                  contacts: np.ndarray):
    """Find the contact points between two cubes."""

    # allocating these arrays of known size allows numba to allocate
    # them on the stack, which is faster than heap allocation
    a_proj = np.empty(8, np.float64)
    b_proj = np.empty(8, np.float64)

    a_idx = np.empty(8, np.int64)
    b_idx = np.empty(8, np.int64)

    # Project the vertices of the cubes onto the separating axis
    a_proj[:] = a_vertices.dot(normal)
    b_proj[:] = b_vertices.dot(normal)

    # the separating plane will be perpendicular to the normal
    # and intersect at the point between the maximum of a and the
    # minimum of b
    a_max = np.max(a_proj)
    b_min = np.min(b_proj)
    midpoint = (a_max + b_min) * 0.5

    # We do not want to consider all the vertices of the cubes, only
    # the ones that are close to the separating plane.
    THRESHOLD = 0.05

    num_a = 0
    for i in range(8):
        if midpoint - a_proj[i] < THRESHOLD:
            a_idx[num_a] = i
            num_a += 1

    if num_a == 0:
        return 0

    num_b = 0
    for i in range(8):
        if b_proj[i] - midpoint < THRESHOLD:
            b_idx[num_b] = i
            num_b += 1

    if num_b == 0:
        return 0

    # we will store contacts in this array. As we need to eliminate
    # duplicates, we will store them as integers and divide by TOL
    # to get the floating point values once we are done
    num_contacts = 0
    unique_contacts = np.empty((12, 3), dtype=np.int64)
    min_dist = np.inf

    for i in range(num_a):
        num_contacts, min_dist = find_contact_point_cube(a_vertices[a_idx[i]],
                                                         b_centroid, b_size, b_rotation,
                                                         unique_contacts, min_dist,
                                                         num_contacts)

    for i in range(num_b):
        num_contacts, min_dist = find_contact_point_cube(b_vertices[b_idx[i]],
                                                         a_centroid, a_size, a_rotation,
                                                         unique_contacts, min_dist,
                                                         num_contacts)

    # now we need to find any edge intersections
    a_edges = np.empty((12, 2), dtype=np.int64)
    num_a_edges = find_edges(a_idx, num_a, CUBE_EDGE_INDEX, a_edges)
    if num_a_edges == 0:
        contacts[:num_contacts] = unique_contacts[:num_contacts] * TOL
        return num_contacts

    b_edges = np.empty((12, 2), dtype=np.int64)
    num_b_edges = find_edges(b_idx, num_b, CUBE_EDGE_INDEX, b_edges)
    if num_b_edges == 0:
        contacts[:num_contacts] = unique_contacts[:num_contacts] * TOL
        return num_contacts

    num_contacts, min_dist = find_contacts_edges(a_vertices, num_a_edges, a_edges,
                                                 b_vertices, num_b_edges, b_edges,
                                                 unique_contacts, min_dist, num_contacts)

    # we need to convert the integer contacts to floating point
    contacts[:num_contacts] = unique_contacts[:num_contacts] * TOL
    return num_contacts


def separate_bodies(a: RigidBody, b: RigidBody, collision: Collision):
    if a.physics and b.physics:
        adjust = (collision.depth * 0.5) * collision.normal
        a.move(-adjust)
        b.move(adjust)
    elif a.physics:
        a.move(-collision.normal * collision.depth)
    elif b.physics:
        b.move(collision.normal * collision.depth)
    else:
        raise ValueError("Cannot separate bodies (neither is in physics system)")


def find_contact_points(a: RigidBody, b: RigidBody,
                        collision: Collision, contacts: np.ndarray) -> int:
    separate_bodies(a, b, collision)
    normal = collision.normal
    match a.kind, b.kind:
        case PrimitiveKind.SPHERE, PrimitiveKind.SPHERE:
            contacts[0] = a.position + normal * a.radius
            return 1
        case PrimitiveKind.SPHERE, _:
            contacts[0] = a.position + normal * a.radius
            return 1
        case _, PrimitiveKind.SPHERE:
            contacts[0] = b.position - normal * b.radius
            return 1
        case PrimitiveKind.CUBOID, PrimitiveKind.CUBOID:
            return find_contact_points_cube_cube(a.position,
                                                 a.transformed_vertices,
                                                 a.size,
                                                 a.rotation,
                                                 b.position,
                                                 b.transformed_vertices,
                                                 b.size,
                                                 b.rotation,
                                                 normal,
                                                 contacts)
        case _:
            raise NotImplementedError(f"Unsupported primitive kind for contacts: {a.kind}, {b.kind}")
