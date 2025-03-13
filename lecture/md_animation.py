"""Script producing an animation of the Minkowski difference of two polygons.

Out of scope for the Tripos.
"""

import random
from typing import List
import cv2
import numpy as np
from pygame import Vector2

from rigidphysics.config import PhysicsMode
from rigidphysics.flat.collisions import detect_collision
from rigidphysics.flat.physics import Physics
from video_writer import VideoWriter
from rigidphysics.flat.bodies import Polygon
from convex_hull import compute_convex_hull


def minkowski_difference(a: List[Vector2], b: List[Vector2]) -> List[Vector2]:
    return [ai - bi for ai in a for bi in b]


def md_animation(width=800, height=900, num_seconds=60, framerate=30, seed=20080524):
    random.seed(seed)
    a = Polygon.create_regular_polygon(3, width / 8, 2, (255, 0, 0))
    b = Polygon.create_regular_polygon(5, width / 8, 2, (0, 255, 0))
    a.linear_velocity = Vector2(random.uniform(-200, 200), random.uniform(-200, 200))
    b.linear_velocity = Vector2(random.uniform(-200, 200), random.uniform(-200, 200))
    a.angular_velocity = random.uniform(-1, 1)
    b.angular_velocity = random.uniform(-1, 1)
    cx, cy = width // 2, height // 2
    center = np.array([cx, cy])
    border = 16
    dt = 1 / framerate
    gravity = Vector2()
    wt = Polygon.create_rectangle(width, border, 2, (0, 0, 0), True).move_to(Vector2(0, border/2-cy))
    wb = Polygon.create_rectangle(width, border, 2, (0, 0, 0), True).move_to(Vector2(0, cy - border / 2))
    wl = Polygon.create_rectangle(border, height, 2, (0, 0, 0), True).move_to(Vector2(border / 2 - cx, 0))
    wr = Polygon.create_rectangle(border, height, 2, (0, 0, 0), True).move_to(Vector2(cx - border / 2, 0))
    walls = [wt, wb, wl, wr]
    physics = Physics(PhysicsMode.BASIC, 1, 0, 0)
    a.physics = True
    b.physics = True
    for w in walls:
        w.physics = False

    def to_curve(points: np.ndarray) -> np.ndarray:
        return [(points + center).reshape(-1, 1, 2).astype(np.int32)]

    with VideoWriter("minkowski_difference.mp4",
                     (width * 2, height),
                     framerate=framerate,
                     quality=17,
                     background_color=(1, 1, 1)) as video:
        left = video.frame[:, :width]
        right = video.frame[:, width:]

        for _ in range(num_seconds * 30):
            video.clear_frame()
            axis_color = (80, 80, 80)
            cv2.line(left, (border, cy), (width-border, cy), axis_color, 2)
            cv2.line(left, (cx, border), (cx, height-border), axis_color, 2)
            cv2.circle(left, (cx, cy), 8, axis_color, -1)
            cv2.line(right, (border, cy), (width - border, cy), axis_color, 2)
            cv2.line(right, (cx, border), (cx, height - border), axis_color, 2)
            cv2.circle(right, (cx, cy), 8, axis_color, -1)
            cv2.line(video.frame, (width, 0), (width, height), (0, 0, 0), 4)

            a_verts = np.array(a.transformed_vertices)
            b_verts = np.array(b.transformed_vertices)
            collision = detect_collision(a, b)
            if collision:
                a_color, b_color = (0, 0, 255), (0, 0, 255)
            else:
                a_color, b_color = a.color, b.color

            cv2.polylines(right, to_curve(a_verts), True, a_color, 8)
            cv2.polylines(right, to_curve(b_verts), True, b_color, 8)

            md = minkowski_difference(a.transformed_vertices, b.transformed_vertices)
            hull = np.array(compute_convex_hull(md))
            cv2.polylines(left, to_curve(hull * 0.5), True, (255, 255, 0), 8)

            cv2.polylines(left, to_curve((a_verts - b.position) * 0.5), True, a_color, 2)
            for v in a_verts:
                cv2.polylines(left, to_curve((v - b_verts) * 0.5), True, b_color, 2)

            a.step(dt, gravity)
            b.step(dt, gravity)

            for w in walls:
                collision = detect_collision(a, w)
                if collision:
                    physics.resolve_collision(a, w, collision, None, None)
                collision = detect_collision(b, w)
                if collision:
                    physics.resolve_collision(b, w, collision, None, None)

            video.write_frame()


if __name__ == "__main__":
    md_animation(num_seconds=60)
