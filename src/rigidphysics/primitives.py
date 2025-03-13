"""Module providing primitive shapes for the graphics engine.

NB This module is out of scope for the Tripos.
"""

import os
from typing import List, NamedTuple, Tuple, Union

import numpy as np
from pyglet.math import Vec2, Vec3
from scipy.spatial.transform import Rotation

from .config import PrimitiveKind


Triangle = Tuple[int, int, int]
Face = NamedTuple("Face", [("vertices", Triangle),
                           ("normals", Triangle),
                           ("uvs", Triangle)])


Mesh = NamedTuple("Mesh", [("vertices", List[Vec3]),
                           ("normals", List[Vec3]),
                           ("uvs", List[Vec2]),
                           ("faces", List[Face])])


def load_mesh(name: str):
    obj_path = os.path.join(os.path.dirname(__file__), "assets", name)
    if not obj_path.endswith(".obj"):
        obj_path += ".obj"

    vertices = []
    uvs = []
    normals = []
    faces = []
    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            parts = line.split()
            match parts[0]:
                case "v":
                    vertices.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                case "vn":
                    normals.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                case "vt":
                    uvs.append(Vec2(float(parts[1]), float(parts[2])))
                case "f":
                    v_idx = []
                    uv_idx = []
                    n_idx = []
                    for part in parts[1:]:
                        indices = part.split("/")
                        v_idx.append(int(indices[0]) - 1)
                        uv_idx.append(int(indices[1]) - 1)
                        n_idx.append(int(indices[2]) - 1)

                    faces.append(Face(tuple(v_idx), tuple(n_idx), tuple(uv_idx)))

    return Mesh(vertices, normals, uvs, faces)


class Primitive(NamedTuple("Primitive", [("kind", PrimitiveKind),
                                         ("vertices", np.ndarray),
                                         ("radius", float),
                                         ("checkerboard", np.ndarray),
                                         ("buffer", np.ndarray)])):
    ELEMENTS_PER_VERTEX = 3 + 3 + 2

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0] * 3

    def pose(self, positions: np.ndarray,
             scales: np.ndarray, rotations: np.ndarray,
             out: np.ndarray = None):
        num_instances = len(positions)
        num_verts = self.vertices.shape[0]
        rotations = rotations.reshape(-1, 1, 4)
        rotations = np.broadcast_to(rotations, (num_instances, num_verts, 4))
        rotations = Rotation.from_quat(rotations.reshape(-1, 4))

        vertices = self.vertices.reshape(1, -1, 3)
        vertices = np.broadcast_to(vertices, (num_instances, num_verts, 3)).reshape(-1, 3)

        if out is None:
            out = np.zeros((num_instances, num_verts, 3), np.float64)

        out[:] = rotations.apply(vertices).reshape(num_instances, num_verts, 3)
        scales = scales.reshape(num_instances, 1, 3)
        positions = positions.reshape(num_instances, 1, 3)
        out *= scales
        out += positions

        return out

    @staticmethod
    def create(kind: Union[str, PrimitiveKind]) -> "Primitive":
        if isinstance(kind, str):
            mesh = load_mesh(kind)
            kind = PrimitiveKind[kind.upper()]
        else:
            mesh = load_mesh(str(kind))

        unique_vertices = np.array(mesh.vertices, np.float32)
        vertices = []
        normals = []
        uvs = []
        for face in mesh.faces:
            for i in range(3):
                vertices.append(unique_vertices[face.vertices[i]])
                normals.append(mesh.normals[face.normals[i]])
                uvs.append(mesh.uvs[face.uvs[i]])

        vertices = np.array(vertices, np.float32)
        normals = np.array(normals, np.float32)
        uvs = np.array(uvs, np.float32)
        checkerboard = np.ones(2, np.float32)

        if kind == PrimitiveKind.SPHERE:
            radius = 0.5
            checkerboard[0] = radius * 2 * np.pi
            checkerboard[1] = radius * np.pi
        else:
            radius = np.linalg.norm(unique_vertices - unique_vertices.mean(axis=0), axis=-1).max()

        buffer = np.concatenate([vertices.flatten(), normals.flatten(), uvs.flatten()])
        return Primitive(kind, unique_vertices, radius, checkerboard, buffer)

    def apply(self, rotation: np.ndarray, scale: float) -> "Primitive":
        vertices = self.vertices * scale
        vertices = vertices @ rotation[:3, :3].T
        half = len(self.buffer) // 2
        positions = self.buffer[:half].reshape(-1, 3, 3) * scale
        positions = positions @ rotation[:3, :3].T
        normals = Primitive.flat_normals(positions)
        buffer = np.concatenate([positions.flatten(), normals.flatten()])
        return Primitive(self.kind, self.color, vertices, buffer)


class Lines(NamedTuple("Lines", [("buffer", np.ndarray),
                                 ("num_vertices", int),
                                 ("color", Vec3)])):
    @staticmethod
    def create_cube(color: Vec3, scale: float) -> "Lines":
        vertices = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ], np.float32)

        lines = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ], np.int32)

        num_vertices = len(lines) * 2
        buffer = vertices.take(lines, 0).flatten() * scale
        return Lines(buffer, num_vertices, color)

    @staticmethod
    def create_cursor(color: Vec3, scale: float) -> "Lines":
        vertices = np.array([
            [-0.5, 0, 0],
            [0.5, 0, 0],
            [0, -0.5, 0],
            [0, 0.5, 0],
            [0, 0, -0.5],
            [0, 0, 0.5]
        ], np.float32)

        lines = np.array([
            [0, 1], [2, 3], [4, 5]
        ], np.int32)

        num_vertices = len(lines) * 2
        buffer = vertices.take(lines, 0).flatten() * scale
        return Lines(buffer, num_vertices, color)
