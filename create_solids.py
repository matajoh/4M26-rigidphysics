"""Script to create some basic solids and a viewer for them.

Out of scope for the Tripos.
"""

import math
import os
from typing import List, Mapping, NamedTuple
import numpy as np
import scenepic as sp

from pyglet.math import Vec2, Vec3


class Triangle(NamedTuple("Triangle", [("a", int), ("b", int), ("c", int)])):
    pass


class Edge(NamedTuple("Edge", [("a", int), ("b", int)])):
    pass


class Mesh(NamedTuple("Mesh", [("vertices", List[Vec3]),
                               ("faces", List[Triangle]),
                               ("normals", List[Vec3]),
                               ("uvs", List[Vec2]),
                               ("vertex_based", bool)])):
    def save_obj(self, path: str):
        with open(path, "w") as file:
            for v in self.vertices:
                file.write(f"v {v.x} {v.y} {v.z}\n")

            for vt in self.uvs:
                file.write(f"vt {vt.x} {vt.y}\n")

            for vn in self.normals:
                file.write(f"vn {vn.x} {vn.y} {vn.z}\n")

            for i, (a, b, c) in enumerate(self.faces):
                v1, v2, v3 = a + 1, b + 1, c + 1
                if self.vertex_based:
                    vt1, vt2, vt3 = v1, v2, v3
                    vn1, vn2, vn3 = v1, v2, v3
                else:
                    t = i * 3
                    vt1, vt2, vt3 = t + 1, t + 2, t + 3
                    vn1, vn2, vn3 = i + 1, i + 1, i + 1

                file.write(f"f {v1}/{vt1}/{vn1} {v2}/{vt2}/{vn2} {v3}/{vt3}/{vn3}\n")

    @staticmethod
    def cuboid(scale=1) -> "Mesh":
        v = scale / 2
        vertices = [
            Vec3(-v, -v, -v),
            Vec3(v, -v, -v),
            Vec3(-v, v, -v),
            Vec3(v, v, -v),
            Vec3(-v, -v, v),
            Vec3(v, -v, v),
            Vec3(-v, v, v),
            Vec3(v, v, v)
        ]

        faces = [
            Triangle(0, 2, 3), Triangle(0, 3, 1),
            Triangle(1, 3, 7), Triangle(1, 7, 5),
            Triangle(5, 7, 6), Triangle(5, 6, 4),
            Triangle(4, 6, 2), Triangle(4, 2, 0),
            Triangle(2, 6, 7), Triangle(2, 7, 3),
            Triangle(4, 0, 1), Triangle(4, 1, 5)
        ]

        normals = [
            Vec3(0, 0, -1), Vec3(0, 0, -1),
            Vec3(1, 0, 0), Vec3(1, 0, 0),
            Vec3(0, 0, 1), Vec3(0, 0, 1),
            Vec3(-1, 0, 0), Vec3(-1, 0, 0),
            Vec3(0, 1, 0), Vec3(0, 1, 0),
            Vec3(0, -1, 0), Vec3(0, -1, 0)
        ]

        uvs = [
            Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 0), Vec2(1, 1), Vec2(0, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 0), Vec2(1, 1), Vec2(0, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 0), Vec2(1, 1), Vec2(0, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 0), Vec2(1, 1), Vec2(0, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 0), Vec2(1, 1), Vec2(0, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 0), Vec2(1, 1), Vec2(0, 1),
        ]

        return Mesh(vertices, faces, normals, uvs, False)

    @staticmethod
    def sphere(radius=0.5, steps=2) -> "Mesh":
        t = 0.5 * (1.0 + math.sqrt(5.0))
        vertices = [
            Vec3(-1, t, 0).normalize() * radius,
            Vec3(1, t, 0).normalize() * radius,
            Vec3(-1, -t, 0).normalize() * radius,
            Vec3(1, -t, 0).normalize() * radius,
            Vec3(0, -1, t).normalize() * radius,
            Vec3(0, 1, t).normalize() * radius,
            Vec3(0, -1, -t).normalize() * radius,
            Vec3(0, 1, -t).normalize() * radius,
            Vec3(t, 0, -1).normalize() * radius,
            Vec3(t, 0, 1).normalize() * radius,
            Vec3(-t, 0, -1).normalize() * radius,
            Vec3(-t, 0, 1).normalize() * radius
        ]
        triangles = [
            Triangle(0, 11, 5),
            Triangle(0, 5, 1),
            Triangle(0, 1, 7),
            Triangle(0, 7, 10),
            Triangle(0, 10, 11),
            Triangle(1, 5, 9),
            Triangle(5, 11, 4),
            Triangle(11, 10, 2),
            Triangle(10, 7, 6),
            Triangle(7, 1, 8),
            Triangle(3, 9, 4),
            Triangle(3, 4, 2),
            Triangle(3, 2, 6),
            Triangle(3, 6, 8),
            Triangle(3, 8, 9),
            Triangle(4, 9, 5),
            Triangle(2, 4, 11),
            Triangle(6, 2, 10),
            Triangle(8, 6, 7),
            Triangle(9, 8, 1)
        ]

        for _ in range(steps):
            new_e_vs: Mapping[Edge, int] = {}
            new_triangles: List[Triangle] = []

            for a, b, c in triangles:
                v_idx: Mapping[Edge, int] = {}

                edges = [Edge(a, b), Edge(a, c), Edge(b, c)]
                for edge in edges:
                    backEdge = Edge(edge[1], edge[0])
                    if edge in new_e_vs:
                        v_idx[edge] = new_e_vs[edge]
                    elif backEdge in new_e_vs:
                        v_idx[edge] = new_e_vs[backEdge]
                    else:
                        v = vertices[edge[0]] + vertices[edge[1]]
                        v = v.normalize() * radius
                        v_idx[edge] = new_e_vs[edge] = len(vertices)
                        vertices.append(v)

                new_triangles.append(Triangle(a, v_idx[Edge(a, b)], v_idx[Edge(a, c)]))
                new_triangles.append(Triangle(v_idx[Edge(a, b)], v_idx[Edge(b, c)], v_idx[Edge(a, c)]))
                new_triangles.append(Triangle(v_idx[Edge(a, c)], v_idx[Edge(b, c)], c))
                new_triangles.append(Triangle(v_idx[Edge(a, b)], b, v_idx[Edge(b, c)]))

            triangles = new_triangles

        uvs: List[Vec2] = []
        for vtx in vertices:
            u = 0.5 * (1 - math.atan2(vtx.z, vtx.x) / math.pi)
            v = 1 - math.acos(vtx.y * 2) / math.pi
            uvs.append(Vec2(u, v))

        # duplicate vertices across the longitude seam
        new_triangles = []
        for a, b, c in triangles:
            # ensure winding order is correct
            for _ in range(2):
                if uvs[a].x < uvs[b].x or uvs[a].x < uvs[c].x:
                    a, b, c = b, c, a

            a_is_east = uvs[a].x > 2 / 3
            b_is_west = uvs[b].x < 1 / 3
            c_is_west = uvs[c].x < 1 / 3

            if a_is_east and c_is_west:
                # duplicate c vertex
                new_vertex = vertices[c]
                vertices.append(new_vertex)
                uvs.append(Vec2(1.0 + uvs[c].x, uvs[c].y))
                c = len(vertices) - 1

            if a_is_east and b_is_west:
                # duplicate b vertex
                new_vertex = vertices[b]
                vertices.append(new_vertex)
                uvs.append(Vec2(1.0 + uvs[b].x, uvs[b].y))
                b = len(vertices) - 1

            new_triangles.append(Triangle(a, b, c))

        normals = [v.normalize() for v in vertices]
        triangles = new_triangles
        return Mesh(vertices, triangles, normals, uvs, True)

    @staticmethod
    def tetrahedron(scale=1) -> "Mesh":
        d = scale / 2
        cos30 = math.sqrt(3) * 0.5 * d
        sin30 = 0.5 * d
        vertices: List[Vec3] = [
            Vec3(-cos30, -sin30, -sin30),
            Vec3(0, -sin30, d),
            Vec3(cos30, -sin30, -sin30),
            Vec3(0, d, 0)
        ]
        faces = [
            Triangle(0, 2, 1),
            Triangle(0, 1, 3),
            Triangle(1, 2, 3),
            Triangle(2, 0, 3)
        ]

        normals = [(vertices[t[1]] - vertices[t[0]]).cross(vertices[t[2]] - vertices[t[0]]).normalize()
                   for t in faces]

        uvs = [
            Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1)
        ]

        return Mesh(vertices, faces, normals, uvs, False)

    @staticmethod
    def octahedron(scale=1) -> "Mesh":
        d = scale / 2
        v = math.sqrt(2) * d / 2
        vertices = [
            Vec3(-v, 0, -v),
            Vec3(-v, 0, v),
            Vec3(v, 0, v),
            Vec3(v, 0, -v),
            Vec3(0, d, 0),
            Vec3(0, -d, 0),
        ]

        faces = [
            Triangle(0, 4, 3), Triangle(3, 4, 2),
            Triangle(2, 4, 1), Triangle(1, 4, 0),
            Triangle(0, 3, 5), Triangle(3, 2, 5),
            Triangle(2, 1, 5), Triangle(1, 0, 5)
        ]

        normals = [
            (vertices[t[1]] - vertices[t[0]]).cross(vertices[t[2]] - vertices[t[0]]).normalize()
            for t in faces
        ]

        uvs = [
            Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1),
        ]

        return Mesh(vertices, faces, normals, uvs, False)


def create_solids(assets_path: str):
    meshes = [
        ("cuboid", Mesh.cuboid()),
        ("sphere", Mesh.sphere()),
        ("tetrahedron", Mesh.tetrahedron()),
        ("octahedron", Mesh.octahedron())
    ]

    for name, mesh in meshes:
        mesh.save_obj(os.path.join(assets_path, f"{name}.obj"))


def create_viewer(assets_path: str):
    scene = sp.Scene()
    test_pattern = np.zeros((100, 100, 3), dtype=np.uint8)
    for r in range(10):
        rstart = r * 10
        rend = rstart + 10
        for c in range(10):
            if (r + c) % 2 == 0:
                continue

            cstart = c * 10
            cend = cstart + 10
            test_pattern[rstart:rend, cstart:cend, :] = 255

    square_image = scene.create_image()
    square_image.load(os.path.join("test", "uv_square.png"))
    wide_image = scene.create_image()
    wide_image.load(os.path.join("test", "uv_wide.png"))
    meshes = []
    for name in os.listdir(assets_path):
        if name.endswith(".obj"):
            mesh_info = sp.load_obj(os.path.join(assets_path, name))
            name = name[:-4]

            match name:
                case "cuboid":
                    mesh = scene.create_mesh(name, texture_id=square_image)
                    mesh.add_cube(transform=sp.Transforms.translate([-1, 0, 0]))

                case "sphere":
                    mesh = scene.create_mesh(name, texture_id=wide_image)
                    mesh.add_sphere(transform=sp.Transforms.translate([-1, 0, 0]))

                case _:
                    mesh = scene.create_mesh(name, texture_id=square_image)

            mesh.add_mesh(mesh_info, transform=sp.Transforms.translate([1, 0, 0]))
            meshes.append(mesh)

    canvas = scene.create_canvas_3d(width=800, height=800)
    for m in meshes:
        frame = canvas.create_frame()
        frame.add_mesh(m)

    scene.save_as_html(os.path.join(assets_path, "viewer.html"))


if __name__ == "__main__":
    assets_path = os.path.join(os.path.dirname(__file__), "src", "rigidphysics", "assets")
    if not os.path.exists(assets_path):
        os.makedirs(assets_path)

    create_solids(assets_path)
    create_viewer(assets_path)
