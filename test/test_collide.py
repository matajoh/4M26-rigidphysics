import os
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import scenepic as sp

from rigidphysics.contacts import find_contact_points
from rigidphysics.collisions import collide
from rigidphysics.geometry import CUBE_VERTICES, CUBE_EDGES
from rigidphysics.bodies import Cuboid, Sphere


viz_path = os.path.join(os.path.dirname(__file__), "visualizations", "collide")
if not os.path.exists(viz_path):
    os.makedirs(viz_path)


def cube(cube_size=(1, 1, 1)):
    cube_size = np.array(cube_size, np.float64)
    cube = Cuboid.create(cube_size, 1, False)
    cube.physics = True
    return cube


def sphere(radius=0.5, position=np.zeros(3, np.float64)):
    sphere = Sphere.create(radius, 1)
    sphere.physics = True
    sphere.move_to(position)
    return sphere


contacts_buffer = np.empty((25, 3), np.float64)

# idea behind these tests:
# 1. position cubes/spheres so they intersect
# 2. detect collision (assert)
# 3. separate the objects
# 4. detect no collision (assert)
# 5. non-zero contact points (i.e. they are touching)


def test_sphere_sphere_collide():
    num_azimuth = 36
    num_altitude = 18
    num_tests = num_azimuth * num_altitude
    azimuth = np.linspace(0, 2 * np.pi, num_azimuth)
    altitude = np.linspace(-np.pi/2, np.pi/2, num_altitude)
    radii = np.linspace(0.5, 1.5, num_tests)
    distances = np.linspace(0.1, 0.9, num_tests)
    rows, columns = np.meshgrid(azimuth, altitude, indexing="ij")
    normals = np.zeros((num_azimuth, num_altitude, 3), np.float64)
    normals[..., 0] = np.cos(rows) * np.cos(columns)
    normals[..., 1] = np.sin(rows) * np.cos(columns)
    normals[..., 2] = np.sin(columns)
    normals = normals.reshape(-1, 3)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    for b_normal, b_radius, distance in zip(normals, radii, distances):
        a = sphere()
        b = sphere(b_radius, b_normal * (0.5 * b_radius) * distance)

        collision = collide(a, b)
        assert collision.depth > 0
        np.testing.assert_allclose(collision.normal, b_normal)
        b.move(collision.normal * collision.depth)

        assert collide(a, b) is None


def test_sphere_cube_collide():
    normals = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [1, 1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [-1, -1, 0],
        [1, 0, 1],
        [1, 0, -1],
        [-1, 0, 1],
        [-1, 0, -1],
        [0, 1, 1],
        [0, 1, -1],
        [0, -1, 1],
        [0, -1, -1],
    ]) * 0.5
    normals = np.concatenate([normals, CUBE_VERTICES])
    distances = np.linalg.norm(normals, axis=1)
    normals = normals / distances.reshape(-1, 1)

    a = sphere()
    b = cube()
    for normal, distance in zip(normals, distances):
        b_centroid = 0.9 * normal * (distance + a.radius)
        b.move_to(b_centroid)

        collision = collide(a, b)
        assert collision.depth > 0
        assert np.allclose(collision.normal, normal)
        b.move(collision.normal * collision.depth)

        assert collide(a, b) is None


def test_cube_cube_corner_to_corner():
    positions = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, -1],
        [-1, -1, 1],
    ], np.float64)
    origin = np.zeros(3, np.float64)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)

    a = cube()
    b = cube()
    for pos in positions:
        a_centroid = origin
        b_centroid = pos
        normal = b_centroid - a_centroid
        normal /= np.linalg.norm(normal)
        b_centroid -= 0.1 * normal
        b.move_to(b_centroid)

        transform = sp.Transforms.translate(b_centroid)
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        collision = collide(a, b)

        assert collision is not None
        assert collision.depth > 0
        b.move(collision.normal * collision.depth)

        touching = collide(a, b)
        assert touching is None or touching.depth == 0

    scene.save_as_html(os.path.join(viz_path, "cube_cube_corner_corner.html"))


def test_cube_cube_corner_to_edge():
    rotation = Rotation.from_euler("xyz", [0, np.pi / 4, 0])
    rot_vec = rotation.as_rotvec()

    corners = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ], np.float64)

    corners = rotation.apply(corners).astype(np.float64)

    contacts_list = np.array([
        [0.5, 0.5, 0],
        [0, 0.5, 0.5],
        [-0.5, 0.5, 0],
        [0, 0.5, -0.5],
        [0.5, -0.5, 0],
        [0, -0.5, 0.5],
        [-0.5, -0.5, 0],
        [0, -0.5, -0.5],
    ], np.float64)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)
    rotation = sp.Transforms.rotation_about_y(np.pi / 4)

    a = cube()
    b = cube()
    b.rotate(rot_vec)
    for corner, expected in zip(corners, contacts_list):
        normal = expected.copy()
        normal /= np.linalg.norm(normal)

        b_centroid = expected - 0.9 * corner
        b.move_to(b_centroid)

        transform = sp.Transforms.translate(b_centroid) @ rotation
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        collision = collide(a, b)
        assert collision is not None
        assert collision.depth > 0
        b.move(collision.normal * collision.depth)

        touching = collide(a, b)
        assert touching is None or touching.depth == 0

    scene.save_as_html(os.path.join(viz_path, "cube_cube_corner_edge.html"))


def test_cube_cube_edge_to_edge_inside():
    b_size = np.array([0.5, 0.5, 0.5], np.float64)

    positions = np.array([
        [0.75, 0.75, 0],
        [0, 0.75, 0.75],
        [-0.75, 0.75, 0],
        [0, 0.75, -0.75],
        [0.75, -0.75, 0],
        [0, -0.75, 0.75],
        [-0.75, -0.75, 0],
        [0, -0.75, -0.75],
    ], np.float64)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)
    scale = sp.Transforms.scale(0.5)

    a = cube()
    b = cube(b_size)
    for b_centroid in positions:
        normal = b_centroid
        normal /= np.linalg.norm(normal)

        b_centroid -= 0.1 * normal
        b.move_to(b_centroid)

        transform = sp.Transforms.translate(b_centroid) @ scale
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        collision = collide(a, b)
        assert collision is not None
        assert collision.depth > 0
        b.move(collision.normal * collision.depth)

        touching = collide(a, b)
        assert touching is None or touching.depth == 0

    scene.save_as_html(os.path.join(viz_path, "cube_cube_edge_edge_inside.html"))


def test_cube_cube_edge_to_edge_overlap():
    positions = np.array([
        [1, 1, 0.5],
        [1, 1, -0.5],
        [0.5, 1, 1],
        [-0.5, 1, 1],
        [-1, 1, 0.5],
        [-1, 1, -0.5],
        [0.5, 1, -1],
        [-0.5, 1, -1],
        [1, -1, 0.5],
        [1, -1, -0.5],
        [0.5, -1, 1],
        [-0.5, -1, 1],
        [-1, -1, 0.5],
        [-1, -1, -0.5],
        [0.5, -1, -1],
        [-0.5, -1, -1],
    ], np.float64)

    contacts_list = np.array([
        [[0.5, 0.5, 0], [0.5, 0.5, 0.5]],
        [[0.5, 0.5, 0], [0.5, 0.5, -0.5]],
        [[0, 0.5, 0.5], [0.5, 0.5, 0.5]],
        [[0, 0.5, 0.5], [-0.5, 0.5, 0.5]],
        [[-0.5, 0.5, 0], [-0.5, 0.5, 0.5]],
        [[-0.5, 0.5, 0], [-0.5, 0.5, -0.5]],
        [[0, 0.5, -0.5], [0.5, 0.5, -0.5]],
        [[0, 0.5, -0.5], [-0.5, 0.5, -0.5]],
        [[0.5, -0.5, 0], [0.5, -0.5, 0.5]],
        [[0.5, -0.5, 0], [0.5, -0.5, -0.5]],
        [[0, -0.5, 0.5], [0.5, -0.5, 0.5]],
        [[0, -0.5, 0.5], [-0.5, -0.5, 0.5]],
        [[-0.5, -0.5, 0], [-0.5, -0.5, 0.5]],
        [[-0.5, -0.5, 0], [-0.5, -0.5, -0.5]],
        [[0, -0.5, -0.5], [0.5, -0.5, -0.5]],
        [[0, -0.5, -0.5], [-0.5, -0.5, -0.5]],
    ], np.float64)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)
    a = cube()
    b = cube()

    for b_centroid, expected in zip(positions, contacts_list):
        normal = expected[0].copy()
        normal /= np.linalg.norm(normal)

        b_centroid -= 0.1 * normal
        b.move_to(b_centroid)

        transform = sp.Transforms.translate(b_centroid)
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        collision = collide(a, b)
        assert collision is not None
        assert collision.depth > 0
        b.move(collision.normal * collision.depth)

        touching = collide(a, b)
        assert touching is None or touching.depth == 0

    scene.save_as_html(os.path.join(viz_path, "cube_cube_edge_edge_overlap.html"))


def test_cube_cube_edge_to_face_inside():
    b_size = np.array([0.5, 0.5, 0.5], np.float64)
    rotvecs = np.array([
        [0, np.pi / 4, 0],
        [0, np.pi / 4, 0],
        [0, np.pi / 4, 0],
        [0, np.pi / 4, 0],
        [np.pi / 4, 0, 0],
        [np.pi / 4, 0, 0]
    ], np.float64)

    d = np.sqrt(2 * 0.0625) + 0.5
    positions = np.array([
        [d, 0, 0],
        [0, 0, d],
        [-d, 0, 0],
        [0, 0, -d],
        [0, d, 0],
        [0, -d, 0]
    ], np.float64)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)
    scale = sp.Transforms.scale(0.5)

    a = cube()
    b = cube(b_size)

    for rotvec, b_centroid in zip(rotvecs, positions):
        normal = b_centroid.copy()
        normal /= np.linalg.norm(normal)

        b_centroid -= 0.1 * normal
        b.move_to(b_centroid)
        b.rotate_to(rotvec)

        rotation = sp.Transforms.rotation_about_y(rotvec[1]) @ sp.Transforms.rotation_about_x(rotvec[0])
        transform = sp.Transforms.translate(b_centroid) @ rotation @ scale
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        collision = collide(a, b)
        assert collision is not None
        assert collision.depth > 0
        b.move(collision.normal * collision.depth)

        touching = collide(a, b)
        assert touching is None or touching.depth == 0

    scene.save_as_html(os.path.join(viz_path, "cube_cube_edge_face_inside.html"))


def test_cube_cube_edge_to_face_overlap():
    rotvecs = np.array([
        [0, np.pi / 4, 0],
        [0, np.pi / 4, 0],
        [0, np.pi / 4, 0],
        [0, np.pi / 4, 0],
        [0, np.pi / 4, 0],
        [0, np.pi / 4, 0],
        [0, np.pi / 4, 0],
        [0, np.pi / 4, 0],
        [np.pi / 4, 0, 0],
        [np.pi / 4, 0, 0],
        [np.pi / 4, 0, 0],
        [np.pi / 4, 0, 0]
    ], np.float64)

    d = np.sqrt(.5) + 0.5
    positions = np.array([
        [d, 0.5, 0],
        [d, -0.5, 0],
        [0, 0.5, d],
        [0, -0.5, d],
        [-d, 0.5, 0],
        [-d, -0.5, 0],
        [0, 0.5, -d],
        [0, -0.5, -d],
        [0.5, d, 0],
        [-0.5, d, 0],
        [0.5, -d, 0],
        [-0.5, -d, 0],
    ], np.float64)

    contacts_list = np.array([
        [[0.5, 0, 0], [0.5, 0.5, 0]],
        [[0.5, 0, 0], [0.5, -0.5, 0]],
        [[0, 0, 0.5], [0, 0.5, 0.5]],
        [[0, 0, 0.5], [0, -0.5, 0.5]],
        [[-0.5, 0, 0], [-0.5, 0.5, 0]],
        [[-0.5, 0, 0], [-0.5, -0.5, 0]],
        [[0, 0, -0.5], [0, 0.5, -0.5]],
        [[0, 0, -0.5], [0, -0.5, -0.5]],
        [[0, 0.5, 0], [.5, 0.5, 0]],
        [[0, 0.5, 0], [-.5, 0.5, 0]],
        [[0, -0.5, 0], [.5, -0.5, 0]],
        [[0, -0.5, 0], [-.5, -0.5, 0]]
    ], np.float64)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)

    a = cube()
    b = cube()

    for rotvec, b_centroid, expected in zip(rotvecs, positions, contacts_list):
        normal = expected[0]
        normal /= np.linalg.norm(normal)
        b_centroid -= 0.1 * normal
        b.move_to(b_centroid)
        b.rotate_to(rotvec)

        rotation = Rotation.from_rotvec(rotvec)
        sp_rot = np.eye(4, dtype=np.float64)
        sp_rot[:3, :3] = rotation.as_matrix()
        transform = sp.Transforms.translate(b_centroid) @ sp_rot
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        collision = collide(a, b)
        assert collision is not None
        assert collision.depth > 0
        b.move(collision.normal * collision.depth)

        touching = collide(a, b)
        assert touching is None or touching.depth == 0

    scene.save_as_html(os.path.join(viz_path, "cube_cube_edge_face_overlap.html"))


def test_cube_cube_face_to_face_inside():
    b_size = np.array([0.5, 0.5, 0.5], np.float64)
    d = 0.75
    positions = np.array([
        [d, 0, 0],
        [0, 0, d],
        [-d, 0, 0],
        [0, 0, -d],
        [0, d, 0],
        [0, -d, 0]
    ], np.float64)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)
    scale = sp.Transforms.scale(0.5)

    a = cube()
    b = cube(b_size)

    for b_centroid in positions:
        normal = b_centroid.copy()
        normal /= np.linalg.norm(normal)

        b_centroid -= 0.1 * normal
        b.move_to(b_centroid)

        transform = sp.Transforms.translate(b_centroid) @ scale
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        collision = collide(a, b)
        assert collision is not None
        assert collision.depth > 0
        b.move(collision.normal * collision.depth)

        touching = collide(a, b)
        assert touching is None or touching.depth == 0

    scene.save_as_html(os.path.join(viz_path, "cube_cube_face_face_inside.html"))


def test_cube_cube_face_to_face_overlap_edge():
    b_size = np.array([0.5, 0.5, 0.5], np.float64)

    scene = sp.Scene()
    scale = sp.Transforms.scale(0.5)
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)

    a = cube()
    b = cube(b_size)
    for edge in CUBE_EDGES:
        contact_mid = (CUBE_VERTICES[edge[0]] + CUBE_VERTICES[edge[1]]) / 2
        if contact_mid[0] == 0:
            sign = np.sign(contact_mid + [CUBE_VERTICES[edge[0], 0], 0, 0], dtype=np.float64)
            offsets = np.array([[0, .25, 0],
                                [0, 0, 0.25]], np.float64) * sign
        elif contact_mid[1] == 0:
            sign = np.sign(contact_mid + [0, CUBE_VERTICES[edge[0], 1], 0], dtype=np.float64)
            offsets = np.array([[.25, 0, 0],
                                [0, 0, 0.25]], np.float64) * sign
        else:
            sign = np.sign(contact_mid + [0, 0, CUBE_VERTICES[edge[0], 2]], dtype=np.float64)
            offsets = np.array([[.25, 0, 0],
                                [0, .25, 0]], np.float64) * sign

        positions = contact_mid + offsets
        normals = np.sign(offsets, dtype=np.float64)

        for b_centroid, normal in zip(positions, normals):
            b_centroid -= 0.1 * normal
            b.move_to(b_centroid)

            transform = sp.Transforms.translate(b_centroid) @ scale
            mesh = scene.create_mesh()
            mesh.add_cube(sp.Colors.Blue)
            mesh.add_cube(sp.Colors.Green, transform=transform)
            frame = canvas.create_frame()
            frame.add_mesh(mesh)

            collision = collide(a, b)
            assert collision is not None
            assert collision.depth > 0
            b.move(collision.normal * collision.depth)

            touching = collide(a, b)
            assert touching is None or touching.depth == 0

    scene.save_as_html(os.path.join(viz_path, "cube_cube_face_face_overlap_edge.html"))


def test_cube_cube_face_to_face_overlap_corner():
    positions = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, -1],
        [-1, -1, 1],
    ], np.float64)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)

    a = cube()
    b = cube()
    for start in positions:
        offsets = np.zeros((3, 3), np.float64)
        offsets[:] = start * 0.5
        np.fill_diagonal(offsets, 0)
        positions = start - offsets
        normals = np.diag(np.sign(start))
        contacts_list = np.zeros((3, 4, 3), np.float64)
        for i in range(3):
            contacts_list[i, :, i] = 0.5 * start[i]
            contacts_list[i, :2, (i + 1) % 3] = 0.5 * start[(i + 1) % 3]
            contacts_list[i, 1:3, (i + 2) % 3] = 0.5 * start[(i + 2) % 3]

        for b_centroid, expected, normal in zip(positions, contacts_list, normals):
            b_centroid -= 0.1 * normal
            b.move_to(b_centroid)

            mesh = scene.create_mesh()
            mesh.add_cube(sp.Colors.Blue)
            mesh.add_cube(sp.Colors.Green, transform=sp.Transforms.translate(b_centroid))
            frame = canvas.create_frame()
            frame.add_mesh(mesh)

            collision = collide(a, b)
            assert collision is not None
            assert collision.depth > 0
            b.move(collision.normal * collision.depth)

            touching = collide(a, b)
            assert touching is None or touching.depth == 0

    scene.save_as_html(os.path.join(viz_path, "cube_cube_face_face_overlap_corner.html"))


def test_cube_cube_face_to_face_overlap_cross():
    scales = np.array([
        [1, 2, 0.5],
        [1, 0.5, 2],
        [1, 2, 0.5],
        [1, 0.5, 2],
        [2, 1, 0.5],
        [0.5, 1, 2],
        [2, 1, 0.5],
        [0.5, 1, 2],
        [2, 0.5, 1],
        [0.5, 2, 1],
        [2, 0.5, 1],
        [0.5, 2, 1],
    ], np.float64)

    positions = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, -1],
        [0, 0, -1],
    ], np.float64)

    a = cube()

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)
    for b_centroid, b_size in zip(positions, scales):
        normal = b_centroid

        b = cube(b_size)
        b_centroid -= 0.1 * normal
        b.move_to(b_centroid)

        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=sp.Transforms.translate(b_centroid) @ sp.Transforms.scale(b_size))
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        collision = collide(a, b)
        assert collision is not None
        assert collision.depth > 0
        b.move(collision.normal * collision.depth)

        touching = collide(a, b)
        assert touching is None or touching.depth == 0

    scene.save_as_html(os.path.join(viz_path, "cube_cube_face_face_overlap_cross.html"))


if __name__ == "__main__":
    test_sphere_sphere_collide()
