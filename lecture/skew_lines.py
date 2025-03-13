"""Out of scope for Tripos."""

import numpy as np
import scenepic as sp

from rigidphysics.maths import closest_points


def main():
    edges = np.array(
        [[[-1, -1, -1], [1, 1, 1]],
         [[-1, -1, 1], [1, 0.5, -1]]], np.float64)
    points = np.zeros((2, 3), np.float64)
    closest_points(edges, points)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d(width=800, height=600)
    canvas.shading = sp.Shading(sp.Colors.White)

    cameras = sp.Camera.orbit(60, 4, 1, 0, 0.5, [0, 1, 0], [0, 0, -1],
                              45, 800/600, 0.01, 20.0)

    mesh = scene.create_mesh()
    mesh.add_thickline(sp.Colors.Red, edges[0, 0], edges[0, 1], 0.05, 0.05)
    mesh.add_thickline(sp.Colors.Green, edges[1, 0], edges[1, 1], 0.05, 0.05)
    mesh.add_thickline(sp.Colors.Blue, points[0], points[1], 0.05, 0.05)

    for camera in cameras:
        canvas.create_frame(camera=camera, meshes=[mesh])

    scene.save_as_html("skew_lines.html")


if __name__ == "__main__":
    main()
