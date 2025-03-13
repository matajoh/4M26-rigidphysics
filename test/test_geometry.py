import os
import numpy as np
import scenepic as sp

from rigidphysics.geometry import CUBE_EDGE_INDEX, CUBE_EDGES
from rigidphysics.bodies import Cuboid


viz_path = os.path.join(os.path.dirname(__file__), "visualizations", "geometry")


def test_cube():
    cube = Cuboid.create([1, 1, 1], 1, False)

    scene = sp.Scene()
    edges = scene.create_mesh(layer_id="edges")
    faces = scene.create_mesh(layer_id="faces")

    for edge in CUBE_EDGES:
        a, b = cube.transformed_vertices[edge]
        edges.add_thickline(sp.Colors.White, a, b)

    canvas = scene.create_canvas_3d()
    frame = canvas.create_frame()
    frame.add_mesh(edges)
    frame.add_mesh(faces)

    scene.save_as_html(os.path.join(viz_path, "cube.html"))

    for edge in CUBE_EDGES:
        a, b = edge + 1
        if a < b:
            key = a << 8 | b
        else:
            key = b << 8 | a

        index = np.searchsorted(CUBE_EDGE_INDEX, key)
        assert CUBE_EDGE_INDEX[index] == key
