"""Collision detection functions."""

from typing import List, NamedTuple

import numpy as np
from numba import jit


from .config import DetectionKind
from .collisions import intersect_cubes
from .bodies import RigidBody
from .geometry import CUBE_NORMALS, CUBE_VERTICES


@jit(cache=True)
def int_coord(v: float, spacing: float) -> int:
    return int(v / spacing)


@jit(cache=True)
def hash_coords(xi: int, yi: int, zi: int, num_cells: int) -> int:
    h = (xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481)
    return abs(h) % num_cells


@jit(cache=True)
def hash_pos(pos: np.ndarray, spacing: float, num_cells: int) -> int:
    return hash_coords(int_coord(pos[0], spacing),
                       int_coord(pos[1], spacing),
                       int_coord(pos[2], spacing), num_cells)


@jit(cache=True)
def detect_spatial_hashing(positions: np.ndarray,
                           aabbs: np.ndarray,
                           num_colliders: int,
                           num_static: int,
                           spacing: float,
                           cell_start: np.ndarray,
                           cell_entries: np.ndarray,
                           static_hits: np.ndarray,
                           hash_values: np.ndarray,
                           phantoms: np.ndarray,
                           pairs: np.ndarray):
    # clear cell start
    cell_start[:] = 0
    num_cells = cell_start.shape[0] - 1
    # first we count how many entries are in each cell
    for i in range(num_colliders):
        h = hash_pos(positions[i], spacing, num_cells)
        hash_values[i] = h
        cell_start[h] += 1

    # perform the cumulative sum
    start = 0
    for i in range(num_cells):
        start += cell_start[i]
        cell_start[i] = start

    cell_start[-1] = start

    # populate the cell entries
    for i in range(num_colliders):
        h = hash_values[i]
        # the effect is that we fill from the back. Once all
        # nodes have been placed, the start will be at the
        # beginning of the cell entries.
        cell_start[h] -= 1
        cell_entries[cell_start[h]] = i

    num_detections = 0
    neighbors = [-1, 0, 1]
    num_phantoms = phantoms.shape[0]
    static_hits[:] = 0
    for a_idx in range(num_phantoms, num_colliders):
        x = int_coord(positions[a_idx, 0], spacing)
        y = int_coord(positions[a_idx, 1], spacing)
        z = int_coord(positions[a_idx, 2], spacing)
        a_xmin, a_ymin, a_zmin = aabbs[a_idx, 0]
        a_xmax, a_ymax, a_zmax = aabbs[a_idx, 1]
        a_body_idx = a_idx - num_phantoms + num_static

        # we will have to look at the 3x3x3 cube around the cell
        for dx in neighbors:
            for dy in neighbors:
                for dz in neighbors:
                    h = hash_coords(x + dx, y + dy, z + dz, num_cells)
                    start = cell_start[h]
                    end = cell_start[h + 1]
                    for j in range(start, end):
                        b_idx = cell_entries[j]
                        if a_idx <= b_idx:
                            continue

                        if b_idx < num_phantoms:
                            b_body_idx = phantoms[b_idx]
                            if static_hits[b_body_idx, a_body_idx] == 1:
                                continue
                        else:
                            b_body_idx = b_idx - num_phantoms + num_static

                        if a_xmax < aabbs[b_idx, 0, 0] or aabbs[b_idx, 1, 0] < a_xmin:
                            continue

                        if a_ymax < aabbs[b_idx, 0, 1] or aabbs[b_idx, 1, 1] < a_ymin:
                            continue

                        if a_zmax < aabbs[b_idx, 0, 2] or aabbs[b_idx, 1, 2] < a_zmin:
                            continue

                        pairs[num_detections, 0] = b_body_idx
                        pairs[num_detections, 1] = a_body_idx
                        num_detections += 1
                        if b_body_idx < num_static:
                            static_hits[b_body_idx, a_body_idx] = 1

    return num_detections


@jit(cache=True)
def detect_basic(aabbs: np.ndarray,
                 num_colliders: int,
                 pairs: np.ndarray) -> int:
    num_detections = 0
    for i in range(num_colliders):
        a_xmin, a_ymin, a_zmin = aabbs[i, 0]
        a_xmax, a_ymax, a_zmax = aabbs[i, 1]
        for j in range(0, i):
            if a_ymax < aabbs[j, 0, 1] or aabbs[j, 1, 1] < a_ymin:
                continue

            if a_xmax < aabbs[j, 0, 0] or aabbs[j, 1, 0] < a_xmin:
                continue

            if a_zmax < aabbs[j, 0, 2] or aabbs[j, 1, 2] < a_zmin:
                continue

            pairs[num_detections, 0] = i
            pairs[num_detections, 1] = j
            num_detections += 1

    return num_detections


LargeCollider = NamedTuple("LargeCollider", [("body", RigidBody),
                                             ("phantoms", np.ndarray)])


class Detection:
    def __init__(self, kind: DetectionKind, num_colliders: int, spacing: float):
        self.kind = kind
        self.num_colliders = num_colliders
        self.spacing = spacing
        self.static_colliders = []
        self.large_ids = set()
        self.num_static = 0
        self.num_phantoms = 0
        self.positions = None
        self.aabbs = None
        self.pairs = None
        self.cell_start = None
        self.cell_entries = None
        self.hash_values = None

    def allocate_buffers(self):
        num_colliders = self.num_phantoms + self.num_colliders
        index = 0
        self.positions = np.zeros((num_colliders, 3), dtype=np.float64)
        self.aabbs = np.zeros((num_colliders, 2, 3), dtype=np.float64)
        self.pairs = np.zeros((num_colliders * num_colliders // 2, 2), dtype=np.int64)
        if self.kind == DetectionKind.BASIC:
            for i, body in enumerate(self.static_colliders):
                self.positions[i] = body.position
                self.aabbs[i] = body.aabb
            return

        num_cells = 2 * num_colliders
        self.static_hits = np.zeros((self.num_static, num_colliders), dtype=np.int8)
        self.cell_start = np.zeros(num_cells + 1, dtype=np.int64)
        self.cell_entries = np.zeros(num_colliders, dtype=np.int64)
        self.hash_values = np.zeros(num_colliders, dtype=np.int64)
        self.phantoms = np.zeros(self.num_phantoms, dtype=np.int64)
        for body, phantoms in self.static_colliders:
            for p in phantoms:
                self.positions[index] = p
                self.aabbs[index] = body.aabb
                self.phantoms[index] = body.index
                self.hash_values[index] = hash_pos(p, self.spacing, num_cells)
                index += 1

    def add_static_collider(self, body: RigidBody):
        assert not body.physics, "Not a static object"
        self.num_static += 1
        if self.kind == DetectionKind.BASIC:
            self.num_phantoms += 1
            self.static_colliders.append(body)
            return

        aabb = body.aabb
        x0, y0, z0 = aabb[0]
        x1, y1, z1 = aabb[1]
        a_verts = body.transformed_vertices
        a_normals = body.transformed_normals
        a_centroid = body.position

        b_verts = CUBE_VERTICES * self.spacing
        b_normals = CUBE_NORMALS
        normal = np.empty(3, np.float64)
        phantoms = []
        num_x_vals = int((x1 - x0) / self.spacing)
        if num_x_vals * self.spacing + 1e-5 < x1 - x0:
            num_x_vals += 1

        num_y_vals = int((y1 - y0) / self.spacing)
        if num_y_vals * self.spacing + 1e-5 < y1 - y0:
            num_y_vals += 1

        num_z_vals = int((z1 - z0) / self.spacing)
        if num_z_vals * self.spacing + 1e-5 < z1 - z0:
            num_z_vals += 1

        for i in range(num_x_vals):
            xf = x0 + (i + 0.5) * self.spacing
            for j in range(num_y_vals):
                yf = y0 + (j + 0.5) * self.spacing
                for k in range(num_z_vals):
                    zf = z0 + (k + 0.5) * self.spacing
                    b_centroid = np.array([xf, yf, zf], np.float64)
                    if intersect_cubes(a_centroid, a_verts, a_normals,
                                       b_centroid, b_centroid + b_verts, b_normals,
                                       normal) == -1:
                        continue

                    phantoms.append(b_centroid)

        self.num_phantoms += len(phantoms)
        self.static_colliders.append(LargeCollider(body, np.stack(phantoms)))

    def detect_collisions(self, colliders: List[RigidBody]) -> int:
        """Detect collisions."""
        colliders = colliders[self.num_static:]

        if len(colliders) == 0:
            return 0

        i = self.num_phantoms
        for body in colliders:
            self.positions[i] = body.position
            self.aabbs[i] = body.aabb
            i += 1

        num_colliders = self.num_phantoms + len(colliders)

        match self.kind:
            case DetectionKind.BASIC:
                return detect_basic(self.aabbs, num_colliders, self.pairs)
            case DetectionKind.SPATIAL_HASHING:
                return detect_spatial_hashing(self.positions, self.aabbs, num_colliders, self.num_static,
                                              self.spacing, self.cell_start, self.cell_entries,
                                              self.static_hits, self.hash_values, self.phantoms, self.pairs)
            case _:
                raise ValueError(f"Invalid detection kind: {self.kind}")
