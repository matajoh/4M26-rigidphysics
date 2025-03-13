"""Module providing mathematical functions for rigid body physics.

NB The quaternion maths is optional, otherwise consider this module in
scope.
"""

import math
import numpy as np
from numba import jit


@jit(cache=True)
def quaternion_rotate(q: np.ndarray, v: np.ndarray, out: np.ndarray):
    qv = np.cross(q[:3], v) + q[3] * v
    out[:] = v + 2 * np.cross(q[:3], qv)


@jit(cache=True)
def quaternion_rotate_batch(q: np.ndarray, v: np.ndarray, out: np.ndarray):
    qv = np.empty(3, np.float64)
    for i in range(v.shape[0]):
        qv[:] = np.cross(q[:3], v[i]) + q[3] * v[i]
        out[i] = v[i] + 2 * np.cross(q[:3], qv)


@jit(cache=True)
def quaternion_inverse_rotate(q: np.ndarray, v: np.ndarray, out: np.ndarray):
    qv = np.cross(-q[:3], v) + q[3] * v
    out[:] = v + 2 * np.cross(-q[:3], qv)


@jit(cache=True)
def quaternion_compose(q0: np.ndarray,
                       q1: np.ndarray,
                       out: np.ndarray):
    b, c, d, a = q0
    x, y, z, w = q1
    out[0] = a * x + b * w + c * z - d * y
    out[1] = a * y - b * z + c * w + d * x
    out[2] = a * z + b * y - c * x + d * w
    out[3] = a * w - b * x - c * y - d * z


@jit(cache=True)
def quaternion_from_rotvec(rotvec: np.ndarray, out: np.ndarray):
    angle = np.linalg.norm(rotvec)
    if angle < 1e-8:
        out[3] = 1
        out[:3] = 0
        return

    axis = rotvec / angle
    half_angle = angle * 0.5
    out[3] = math.cos(half_angle)
    out[:3] = math.sin(half_angle) * axis


def quaternion_identity() -> np.ndarray:
    return np.array([0, 0, 0, 1], np.float64)


@jit(cache=True)
def quaternion_to_matrix(q: np.ndarray, out: np.ndarray):
    # pre: q is a unit quaternion
    qi, qj, qk, qr = q
    s2 = 2
    qii = qi * qi
    qjj = qj * qj
    qkk = qk * qk
    qij = qi * qj
    qik = qi * qk
    qjk = qj * qk
    qir = qi * qr
    qjr = qj * qr
    qkr = qk * qr
    out[0, 0] = 1 - s2 * (qjj + qkk)
    out[0, 1] = s2 * (qij - qkr)
    out[0, 2] = s2 * (qik + qjr)
    out[1, 0] = s2 * (qij + qkr)
    out[1, 1] = 1 - s2 * (qii + qkk)
    out[1, 2] = s2 * (qjk - qir)
    out[2, 0] = s2 * (qik - qjr)
    out[2, 1] = s2 * (qjk + qir)
    out[2, 2] = 1 - s2 * (qii + qjj)


@jit(cache=True)
def between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Check if point c is between points a and b."""
    ax = a[0]
    ay = a[1]
    bx = b[0]
    by = b[1]
    cx = c[0]
    cy = c[1]
    if ax != bx:
        return ax <= cx <= bx or bx <= cx <= ax

    return ay <= cy <= by or by <= cy <= ay


@jit(cache=True)
def colinear(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Check if points a, b, and c are colinear."""
    TOL = 0.005
    ab = b - a
    ac = c - a
    area2 = ab[0] * ac[1] - ab[1] * ac[0]
    if area2 > TOL or area2 < -TOL:
        return False

    return True


@jit(cache=True)
def find_edges(index: np.ndarray, num_verts: int,
               edge_idx: np.ndarray, edges: np.ndarray) -> int:
    """Discover which edges the vertices form.

    Descriptions:
        We have filtered out a list of indices based upon the separation
        plane, but we do not know which edges are present in the set.
        This function uses a precomputed edge index to find potential
        edges.
    """
    num_edges = 0
    for i in range(num_verts - 1):
        a = index[i]
        # we store the index of the lower vertex in the upper 8 bits
        key_hi = (a + 1) << 8
        for j in range(i + 1, num_verts):
            b = index[j]
            key = key_hi | (b + 1)
            # this search is O(log n)
            idx = np.searchsorted(edge_idx, key)
            if idx == edge_idx.shape[0] or edge_idx[idx] != key:
                # edge not found
                continue

            # store the edge in the edge array
            edges[num_edges, 0] = a
            edges[num_edges, 1] = b
            num_edges += 1

    return num_edges


@jit(cache=True)
def insert(contacts: np.ndarray, contact: np.ndarray, count: int) -> int:
    """Inserts a contact into the contact set."""
    if count == 0:
        # first contact
        contacts[0] = contact
        return 1

    # insertion sort, checking for uniqueness
    insert = -1
    for i in range(count):
        if np.all(contacts[i] > contact):
            # this is the right place to put this contact
            insert = i
            break

        if np.all(contacts[i] == contact):
            # contact already exists
            return count

    if insert < 0:
        # contact is larger than all existing contacts
        contacts[count] = contact
        return count + 1

    # copy back everything to the right
    for i in range(count, insert, -1):
        contacts[i] = contacts[i - 1]

    # insert the new contact
    contacts[insert] = contact
    return count + 1


@jit(cache=True)
def closest_points(edges: np.ndarray, points: np.ndarray):
    """Find the closest points between two edges.

    Description:
        Two line segments in 3D space are unlikely to intersect, and thus
        are best treated as skew lines:
        https://en.wikipedia.org/wiki/Skew_lines
        The closest points on two skew lines are found by finding the
        shortest line segment between the two lines.
    """
    p1 = edges[0, 0]
    p2 = edges[1, 0]
    d1 = edges[0, 1] - p1
    d2 = edges[1, 1] - p2
    e1 = np.linalg.norm(d1)
    e2 = np.linalg.norm(d2)
    d1 /= e1
    d2 /= e2
    n = np.cross(d1, d2)
    n1 = np.cross(d1, n)
    n2 = np.cross(d2, n)
    denom = np.dot(d1, n2)
    if np.isclose(denom, 0):
        # lines are parallel
        return 0

    s = np.dot(p2 - p1, n2) / denom
    t = np.dot(p1 - p2, n1) / np.dot(d2, n1)
    if s <= 0 or s >= e1 or t <= 0 or t >= e2:
        return 0

    points[0] = p1 + s * d1
    points[1] = p2 + t * d2
    return 1


@jit(cache=True)
def closest_point_on_axis_aligned_cube(p: np.ndarray,
                                       size: np.ndarray,
                                       point: np.ndarray):
    x_min, y_min, z_min = -size * 0.5
    x_max, y_max, z_max = size * 0.5

    point[:] = p

    if point[0] < x_min:
        point[0] = x_min
    elif point[0] > x_max:
        point[0] = x_max

    if point[1] < y_min:
        point[1] = y_min
    elif point[1] > y_max:
        point[1] = y_max

    if point[2] < z_min:
        point[2] = z_min
    elif point[2] > z_max:
        point[2] = z_max


@jit(cache=True)
def closest_point_on_cube(p: np.ndarray,
                          translation: np.ndarray,
                          size: np.ndarray,
                          rotation: np.ndarray,
                          point: np.ndarray):
    """Find the closest point on the cube.

    Description:
        We are going to achieve this by rotating and translating the
        query point into the frame of reference of this cube and then
        finding the closest point using simple axis tests. Then, we
        rotate and translate the point back to the original frame of
        reference.
    """
    point[:] = p - translation
    quaternion_inverse_rotate(rotation, point, point)
    closest_point_on_axis_aligned_cube(point, size, point)
    quaternion_rotate(rotation, point, point)
    point += translation


@jit(cache=True)
def squared_distance_to_cube(p: np.ndarray,
                             translation: np.ndarray,
                             size: np.ndarray,
                             rotation: np.ndarray) -> float:
    """Find minimum distance to the cube.

    Description:
        We are going to achieve this by rotating and translating the
        query point into the frame of reference of this cube and then
        finding the closest point using simple axis tests.
    """
    p_t = np.empty(3, np.float64)
    q = np.empty(3, np.float64)
    p_t[:] = p - translation
    quaternion_inverse_rotate(rotation, p_t, p_t)
    closest_point_on_axis_aligned_cube(p_t, size, q)
    q -= p_t
    return np.dot(q, q)
