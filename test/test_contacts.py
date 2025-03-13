import os
import numpy as np
from scipy.spatial.transform import Rotation
import scenepic as sp

from rigidphysics.config import RigidBodyConfig
from rigidphysics.contacts import find_contact_points_cube_cube
from rigidphysics.geometry import CUBE_EDGES
from rigidphysics.bodies import Cuboid


viz_path = os.path.join(os.path.dirname(__file__), "visualizations", "contacts")
if not os.path.exists(viz_path):
    os.makedirs(viz_path)

config = RigidBodyConfig.from_dict({
    "name": "cuboid",
    "position": 0,
    "size": 1,
    "velocity": 0,
    "angular_velocity": 0
})


cube_size = np.array([1, 1, 1], np.float64)
cube = Cuboid.create(cube_size, 1, False)
cube_verts = cube.vertices
cube_normals = cube.normals
cube_rotation = cube.rotation


contacts_buffer = np.empty((25, 3), np.float64)


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
    face_normal = np.array([0, 1, 0], np.float64)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)

    for pos in positions:
        a_centroid = origin
        b_centroid = pos
        b_verts = cube_verts + b_centroid
        expected = (a_centroid + b_centroid) / 2
        normal = b_centroid - a_centroid
        normal /= np.linalg.norm(normal)

        transform = sp.Transforms.translate(b_centroid)
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                     b_centroid, b_verts, cube_size, cube_rotation,
                                                     normal, contacts_buffer)
        assert num_contacts == 1
        actual = contacts_buffer[0]
        assert np.allclose(actual, expected), "a/b/corner normal"

        num_contacts = find_contact_points_cube_cube(b_centroid, b_verts, cube_size, cube_rotation,
                                                     a_centroid, cube_verts, cube_size, cube_rotation,
                                                     -normal, contacts_buffer)
        assert num_contacts == 1
        actual = contacts_buffer[0]
        assert np.allclose(actual, expected), "b/a/corner normal"

        if face_normal.dot(normal) < 0:
            face_normal = -face_normal

        num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                     b_centroid, b_verts, cube_size, cube_rotation,
                                                     face_normal, contacts_buffer)
        assert num_contacts == 1
        actual = contacts_buffer[0]
        assert np.allclose(actual, expected), "a/b/face normal"

    scene.save_as_html(os.path.join(viz_path, "cube_cube_corner_corner.html"))


def test_cube_cube_corner_to_edge():
    rotation = Rotation.from_euler("xyz", [0, np.pi / 4, 0])
    rotated_verts = rotation.apply(cube_verts).astype(np.float64)

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
    face_normal = np.array([0, 1, 0], np.float64)

    for corner, expected in zip(corners, contacts_list):
        a_centroid = np.zeros(3, np.float64)
        b_centroid = expected - corner
        b_verts = b_centroid + rotated_verts
        normal = expected - a_centroid
        normal /= np.linalg.norm(normal)

        transform = sp.Transforms.translate(b_centroid) @ rotation
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                     b_centroid, b_verts, cube_size, cube_rotation,
                                                     normal, contacts_buffer)
        assert num_contacts == 1
        actual = contacts_buffer[0]
        assert np.allclose(actual, expected), "a/b/edge normal"

        num_contacts = find_contact_points_cube_cube(b_centroid, b_verts, cube_size, cube_rotation,
                                                     a_centroid, cube_verts, cube_size, cube_rotation,
                                                     -normal, contacts_buffer)
        assert num_contacts == 1
        actual = contacts_buffer[0]
        assert np.allclose(actual, expected), "b/a/edge normal"

        if face_normal.dot(normal) < 0:
            face_normal = -face_normal

        num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                     b_centroid, b_verts, cube_size, cube_rotation,
                                                     face_normal, contacts_buffer)
        assert num_contacts == 1
        actual = contacts_buffer[0]
        assert np.allclose(actual, expected), "a/b/face normal"

    scene.save_as_html(os.path.join(viz_path, "cube_cube_corner_edge.html"))


def test_cube_cube_edge_to_edge_inside():
    scaled_verts = 0.5 * cube_verts
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

    contacts_list = np.array([
        [[0.5, 0.5, 0.25], [0.5, 0.5, -0.25]],
        [[0.25, 0.5, 0.5], [-0.25, 0.5, 0.5]],
        [[-0.5, 0.5, 0.25], [-0.5, 0.5, -0.25]],
        [[0.25, 0.5, -0.5], [-0.25, 0.5, -0.5]],
        [[0.5, -0.5, 0.25], [0.5, -0.5, -0.25]],
        [[-0.25, -0.5, 0.5], [0.25, -0.5, 0.5]],
        [[-0.5, -0.5, 0.25], [-0.5, -0.5, -0.25]],
        [[-0.25, -0.5, -0.5], [0.25, -0.5, -0.5]],
    ], np.float64)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)
    scale = sp.Transforms.scale(0.5)
    contact_scale = sp.Transforms.scale(0.1)
    face_normal = np.array([0, 1, 0], np.float64)

    for b_centroid, expected in zip(positions, contacts_list):
        a_centroid = np.zeros(3, np.float64)
        b_verts = b_centroid + scaled_verts
        normal = b_centroid - a_centroid
        normal /= np.linalg.norm(normal)

        transform = sp.Transforms.translate(b_centroid) @ scale
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        mesh.add_sphere(sp.Colors.Red, transform=sp.Transforms.translate(expected[0]) @ contact_scale)
        mesh.add_sphere(sp.Colors.Red, transform=sp.Transforms.translate(expected[1]) @ contact_scale)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                     b_centroid, b_verts, b_size, cube_rotation,
                                                     normal, contacts_buffer)
        assert num_contacts == 2
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        expected = np.sort(expected, axis=0)
        assert np.allclose(actual, expected), "a/b/edge_normal"

        num_contacts = find_contact_points_cube_cube(b_centroid, b_verts, b_size, cube_rotation,
                                                     a_centroid, cube_verts, cube_size, cube_rotation,
                                                     -normal, contacts_buffer)
        assert num_contacts == 2
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        assert np.allclose(actual, expected), "b/a/edge_normal"

        if face_normal.dot(normal) < 0:
            face_normal = -face_normal

        num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                     b_centroid, b_verts, b_size, cube_rotation,
                                                     face_normal, contacts_buffer)
        assert num_contacts == 2
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        assert np.allclose(actual, expected) or np.allclose(actual[::-1], expected), "a/b/face_normal"

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
    contact_scale = sp.Transforms.scale(0.1)
    face_normal = np.array([0, 1, 0], np.float64)

    for b_centroid, expected in zip(positions, contacts_list):
        a_centroid = np.zeros(3, np.float64)
        b_verts = b_centroid + cube_verts
        normal = expected[0] - a_centroid
        normal /= np.linalg.norm(normal)

        transform = sp.Transforms.translate(b_centroid)
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        mesh.add_sphere(sp.Colors.Red, transform=sp.Transforms.translate(expected[0]) @ contact_scale)
        mesh.add_sphere(sp.Colors.Red, transform=sp.Transforms.translate(expected[1]) @ contact_scale)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                     b_centroid, b_verts, cube_size, cube_rotation,
                                                     normal, contacts_buffer)
        assert num_contacts == 2
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        expected = np.sort(expected, axis=0)
        assert np.allclose(actual, expected), f"a/b/edge_normal: {actual, expected}"

        num_contacts = find_contact_points_cube_cube(b_centroid, b_verts, cube_size, cube_rotation,
                                                     a_centroid, cube_verts, cube_size, cube_rotation,
                                                     -normal, contacts_buffer)
        assert num_contacts == 2
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        assert np.allclose(actual, expected), f"b/a/edge_normal, {actual, expected}"

        if face_normal.dot(normal) < 0:
            face_normal = -face_normal

        num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                     b_centroid, b_verts, cube_size, cube_rotation,
                                                     face_normal, contacts_buffer)
        assert num_contacts == 2
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        assert np.allclose(actual, expected) or np.allclose(actual[::-1], expected), "a/b/face_normal"

    scene.save_as_html(os.path.join(viz_path, "cube_cube_edge_edge_overlap.html"))


def test_cube_cube_edge_to_face_inside():
    scaled_verts = 0.5 * cube_verts
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

    contacts_list = np.array([
        [[0.5, .25, 0], [0.5, -0.25, 0]],
        [[0, .25, 0.5], [0, -0.25, 0.5]],
        [[-0.5, 0.25, 0], [-0.5, -0.25, 0]],
        [[0, 0.25, -0.5], [0, -0.25, -0.5]],
        [[.25, 0.5, 0], [-.25, 0.5, 0]],
        [[-.25, -0.5, 0], [.25, -0.5, 0]]
    ], np.float64)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)
    contact_scale = sp.Transforms.scale(0.1)
    scale = sp.Transforms.scale(0.5)

    for rotvec, b_centroid, expected in zip(rotvecs, positions, contacts_list):
        rotation = Rotation.from_rotvec(rotvec)
        rotated_verts = rotation.apply(scaled_verts).astype(np.float64)
        b_rotation = rotation.as_quat()
        a_centroid = np.zeros(3, np.float64)
        normal = b_centroid - a_centroid
        normal /= np.linalg.norm(normal)
        b_verts = b_centroid + rotated_verts

        rotation = sp.Transforms.rotation_about_y(rotvec[1]) @ sp.Transforms.rotation_about_x(rotvec[0])
        transform = sp.Transforms.translate(b_centroid) @ rotation @ scale
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        mesh.add_sphere(sp.Colors.Red, transform=sp.Transforms.translate(expected[0]) @ contact_scale)
        mesh.add_sphere(sp.Colors.Red, transform=sp.Transforms.translate(expected[1]) @ contact_scale)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                     b_centroid, b_verts, b_size, b_rotation,
                                                     normal, contacts_buffer)
        assert num_contacts == 2
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        expected = np.sort(expected, axis=0)
        assert np.allclose(actual, expected), "a/b/face_normal"

        num_contacts = find_contact_points_cube_cube(b_centroid, b_verts, b_size, b_rotation,
                                                     a_centroid, cube_verts, cube_size, cube_rotation,
                                                     -normal, contacts_buffer)
        assert num_contacts == 2
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        assert np.allclose(actual, expected), "b/a/face_normal"

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
    contact_scale = sp.Transforms.scale(0.1)

    for rotvec, b_centroid, expected in zip(rotvecs, positions, contacts_list):
        rotation = Rotation.from_rotvec(rotvec)
        rotated_verts = rotation.apply(cube_verts).astype(np.float64)
        a_centroid = np.zeros(3, np.float64)
        normal = expected[0] - a_centroid
        normal /= np.linalg.norm(normal)
        b_verts = b_centroid + rotated_verts

        sp_rot = np.eye(4, dtype=np.float64)
        sp_rot[:3, :3] = rotation.as_matrix()
        transform = sp.Transforms.translate(b_centroid) @ sp_rot
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        mesh.add_sphere(sp.Colors.Red, transform=sp.Transforms.translate(expected[0]) @ contact_scale)
        mesh.add_sphere(sp.Colors.Red, transform=sp.Transforms.translate(expected[1]) @ contact_scale)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                     b_centroid, b_verts, cube_size, cube_rotation,
                                                     normal, contacts_buffer)
        assert num_contacts == 2
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        expected = np.sort(expected, axis=0)
        assert np.allclose(actual, expected, atol=1e-3), "a/b/face_normal"

        num_contacts = find_contact_points_cube_cube(b_centroid, b_verts, cube_size, cube_rotation,
                                                     a_centroid, cube_verts, cube_size, cube_rotation,
                                                     -normal, contacts_buffer)
        assert num_contacts == 2
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        assert np.allclose(actual, expected, atol=1e-3), "b/a/face_normal"

    scene.save_as_html(os.path.join(viz_path, "cube_cube_edge_face_overlap.html"))


def test_cube_cube_face_to_face_inside():
    scaled_verts = 0.5 * cube_verts
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

    contacts_list = np.array([
        [[0.5, .25, -.25],
         [0.5, -0.25, -.25],
         [0.5, -0.25, .25],
         [0.5, .25, .25]],
        [[0.25, -.25, .5],
         [-0.25, -0.25, .5],
         [-0.25, 0.25, .5],
         [0.25, .25, .5]],
        [[-0.5, .25, -.25],
         [-0.5, 0.25, .25],
         [-0.5, -0.25, .25],
         [-0.5, -.25, -.25]],
        [[0.25, -.25, -.5],
         [0.25, 0.25, -.5],
         [-0.25, 0.25, -.5],
         [-0.25, -.25, -.5]],
        [[0.25, 0.5, -0.25],
         [0.25, 0.5, 0.25],
         [-0.25, 0.5, 0.25],
         [-0.25, 0.5, -0.25]],
        [[-0.25, -0.5, -0.25],
         [-0.25, -0.5, 0.25],
         [0.25, -0.5, 0.25],
         [0.25, -0.5, -0.25]],
    ], np.float64)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)
    contact_scale = sp.Transforms.scale(0.1)
    scale = sp.Transforms.scale(0.5)

    for b_centroid, expected in zip(positions, contacts_list):
        a_centroid = np.zeros(3, np.float64)
        normal = b_centroid - a_centroid
        normal /= np.linalg.norm(normal)
        b_verts = b_centroid + scaled_verts

        transform = sp.Transforms.translate(b_centroid) @ scale
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=transform)
        for contact in expected:
            mesh.add_sphere(sp.Colors.Red, transform=sp.Transforms.translate(contact) @ contact_scale)
        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                     b_centroid, b_verts, b_size, cube_rotation,
                                                     normal, contacts_buffer)
        assert num_contacts == 4
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        expected = np.sort(expected, axis=0)
        assert np.allclose(actual, expected), "a/b/face_normal"

        num_contacts = find_contact_points_cube_cube(b_centroid, b_verts, b_size, cube_rotation,
                                                     a_centroid, cube_verts, cube_size, cube_rotation,
                                                     -normal, contacts_buffer)
        assert num_contacts == 4
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        assert np.allclose(actual, expected), "b/a/face_normal"

    scene.save_as_html(os.path.join(viz_path, "cube_cube_face_face_inside.html"))


def test_cube_cube_face_to_face_overlap_edge():
    scaled_verts = 0.5 * cube_verts
    b_size = np.array([0.5, 0.5, 0.5], np.float64)

    scene = sp.Scene()
    scale = sp.Transforms.scale(0.5)
    contact_scale = sp.Transforms.scale(0.1)
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)

    for edge in CUBE_EDGES:
        contact_mid = (cube_verts[edge[0]] + cube_verts[edge[1]]) / 2
        if contact_mid[0] == 0:
            sign = np.sign(contact_mid + [cube_verts[edge[0], 0], 0, 0], dtype=np.float64)
            offsets = np.array([[0, .25, 0],
                                [0, 0, 0.25]], np.float64) * sign
            contact_offsets = np.array([
                [[-.25, 0, 0.25],
                 [0.25, 0, 0.25],
                 [0.25, 0, 0],
                 [-.25, 0, 0]],
                [[-.25, 0.25, 0],
                 [.25, 0.25, 0],
                 [.25, 0, 0],
                 [-.25, 0, 0]]], np.float64) * sign
        elif contact_mid[1] == 0:
            sign = np.sign(contact_mid + [0, cube_verts[edge[0], 1], 0], dtype=np.float64)
            offsets = np.array([[.25, 0, 0],
                                [0, 0, 0.25]], np.float64) * sign
            contact_offsets = np.array([
                [[0, -.25, 0.25],
                 [0, 0.25, 0.25],
                 [0, 0.25, 0],
                 [0, -0.25, 0]],
                [[0.25, -0.25, 0],
                 [0.25, .25, 0],
                 [0, .25, 0],
                 [0, -0.25, 0]]], np.float64) * sign
        else:
            sign = np.sign(contact_mid + [0, 0, cube_verts[edge[0], 2]], dtype=np.float64)
            offsets = np.array([[.25, 0, 0],
                                [0, .25, 0]], np.float64) * sign
            contact_offsets = np.array([
                [[0, 0, 0.25],
                 [0, 0.25, 0.25],
                 [0, 0.25, -0.25],
                 [0, 0, -0.25]],
                [[0, 0, 0.25],
                 [0.25, 0, 0.25],
                 [0.25, 0, -0.25],
                 [0, 0, -0.25]]], np.float64) * sign

        positions = contact_mid + offsets
        contacts_list = contact_mid - contact_offsets
        normals = np.sign(offsets, dtype=np.float64)

        for position, expected, normal in zip(positions, contacts_list, normals):
            a_centroid = np.zeros(3, np.float64)
            b_centroid = position
            b_verts = b_centroid + scaled_verts

            transform = sp.Transforms.translate(b_centroid) @ scale
            mesh = scene.create_mesh()
            mesh.add_cube(sp.Colors.Blue)
            mesh.add_cube(sp.Colors.Green, transform=transform)
            for contact in expected:
                mesh.add_sphere(sp.Colors.Red, transform=sp.Transforms.translate(contact) @ contact_scale)

            frame = canvas.create_frame()
            frame.add_mesh(mesh)

            num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                         b_centroid, b_verts, b_size, cube_rotation,
                                                         normal, contacts_buffer)

            assert num_contacts == 4, contacts_buffer[:num_contacts]
            actual = contacts_buffer[:num_contacts]
            actual = np.sort(actual, axis=0)
            expected = np.sort(expected, axis=0)
            assert np.allclose(actual, expected), f"a/b/face_normal: {actual, expected}"

            num_contacts = find_contact_points_cube_cube(b_centroid, b_verts, b_size, cube_rotation,
                                                         a_centroid, cube_verts, cube_size, cube_rotation,
                                                         -normal, contacts_buffer)
            assert num_contacts == 4
            actual = contacts_buffer[:num_contacts]
            actual = np.sort(actual, axis=0)
            assert np.allclose(actual, expected), "b/a/face_normal"

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
    contact_scale = sp.Transforms.scale(0.1)

    a_centroid = np.zeros(3, np.float64)
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
            mesh = scene.create_mesh()
            mesh.add_cube(sp.Colors.Blue)
            mesh.add_cube(sp.Colors.Green, transform=sp.Transforms.translate(b_centroid))
            for contact in expected:
                mesh.add_sphere(sp.Colors.Red, transform=sp.Transforms.translate(contact) @ contact_scale)

            frame = canvas.create_frame()
            frame.add_mesh(mesh)

            b_verts = b_centroid + cube_verts

            num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                         b_centroid, b_verts, cube_size, cube_rotation,
                                                         normal, contacts_buffer)
            assert num_contacts == 4
            actual = contacts_buffer[:num_contacts]
            actual = np.sort(actual, axis=0)
            expected = np.sort(expected, axis=0)
            assert np.allclose(actual, expected)

            num_contacts = find_contact_points_cube_cube(b_centroid, b_verts, cube_size, cube_rotation,
                                                         a_centroid, cube_verts, cube_size, cube_rotation,
                                                         -normal, contacts_buffer)
            assert num_contacts == 4
            actual = contacts_buffer[:num_contacts]
            actual = np.sort(actual, axis=0)
            assert np.allclose(actual, expected)

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

    contacts_list = np.array([
        [[0.5, 0.5, -0.25],
         [0.5, -0.5, -0.25],
         [0.5, -0.5, 0.25],
         [0.5, 0.5, 0.25]],
        [[0.5, 0.25, -0.5],
         [0.5, -0.25, -0.5],
         [0.5, -0.25, 0.5],
         [0.5, 0.25, 0.5]],
        [[-0.5, 0.5, -0.25],
         [-0.5, -0.5, -0.25],
         [-0.5, -0.5, 0.25],
         [-0.5, 0.5, 0.25]],
        [[-0.5, 0.25, -0.5],
         [-0.5, -0.25, -0.5],
         [-0.5, -0.25, 0.5],
         [-0.5, 0.25, 0.5]],
        [[0.5, 0.5, -0.25],
         [-0.5, 0.5, -0.25],
         [-0.5, 0.5, 0.25],
         [0.5, 0.5, 0.25]],
        [[0.25, 0.5, -0.5],
         [-0.25, 0.5, -0.5],
         [-0.25, 0.5, 0.5],
         [0.25, 0.5, 0.5]],
        [[0.5, -0.5, -0.25],
         [-0.5, -0.5, -0.25],
         [-0.5, -0.5, 0.25],
         [0.5, -0.5, 0.25]],
        [[0.25, -0.5, -0.5],
         [-0.25, -0.5, -0.5],
         [-0.25, -0.5, 0.5],
         [0.25, -0.5, 0.5]],
        [[0.5, -0.25, 0.5],
         [-0.5, -0.25, 0.5],
         [-0.5, 0.25, 0.5],
         [0.5, 0.25, 0.5]],
        [[0.25, -0.5, 0.5],
         [-0.25, -0.5, 0.5],
         [-0.25, 0.5, 0.5],
         [0.25, 0.5, 0.5]],
        [[0.5, -0.25, -0.5],
         [-0.5, -0.25, -0.5],
         [-0.5, 0.25, -0.5],
         [0.5, 0.25, -0.5]],
        [[0.25, -0.5, -0.5],
         [-0.25, -0.5, -0.5],
         [-0.25, 0.5, -0.5],
         [0.25, 0.5, -0.5]],
    ], np.float64)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d()
    canvas.shading = sp.Shading(sp.Colors.White)
    contact_scale = sp.Transforms.scale(0.1)
    for b_centroid, scale, expected in zip(positions, scales, contacts_list):
        a_centroid = np.zeros(3, np.float64)
        normal = b_centroid
        b_verts = b_centroid + scale * cube_verts
        b_size = scale

        mesh = scene.create_mesh(layer_id="expected")
        mesh.add_cube(sp.Colors.Blue)
        mesh.add_cube(sp.Colors.Green, transform=sp.Transforms.translate(b_centroid) @ sp.Transforms.scale(scale))
        for contact in expected:
            mesh.add_sphere(sp.Colors.Red, transform=sp.Transforms.translate(contact) @ contact_scale)

        frame = canvas.create_frame()
        frame.add_mesh(mesh)

        num_contacts = find_contact_points_cube_cube(a_centroid, cube_verts, cube_size, cube_rotation,
                                                     b_centroid, b_verts, b_size, cube_rotation,
                                                     normal, contacts_buffer)

        assert num_contacts == 4
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        expected = np.sort(expected, axis=0)
        assert np.allclose(actual, expected)

        num_contacts = find_contact_points_cube_cube(b_centroid, b_verts, b_size, cube_rotation,
                                                     a_centroid, cube_verts, cube_size, cube_rotation,
                                                     -normal, contacts_buffer)
        assert num_contacts == 4
        actual = contacts_buffer[:num_contacts]
        actual = np.sort(actual, axis=0)
        assert np.allclose(actual, expected)

    scene.save_as_html(os.path.join(viz_path, "cube_cube_face_face_overlap_cross.html"))


if __name__ == "__main__":
    test_cube_cube_face_to_face_overlap_edge()
