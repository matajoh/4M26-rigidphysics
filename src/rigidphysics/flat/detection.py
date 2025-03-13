

from typing import List, Tuple
from .bodies import AABB, RigidBody
from ..config import DetectionKind
from .quadtree import QuadTree


Collisions = List[Tuple[RigidBody, RigidBody]]


class Detection:
    def __init__(self, kind: DetectionKind, box: AABB):
        self.kind = kind
        self.box = box

    def find_all_intersections(self, bodies: List[RigidBody], collisions: Collisions):
        match self.kind:
            case DetectionKind.QUADTREE:
                return self.find_all_intersections_quadtree(bodies, collisions)
            case DetectionKind.BASIC:
                return self.find_all_intersections_basic(bodies, collisions)
            case _:
                raise ValueError("Invalid detection kind")

    def find_all_intersections_quadtree(self, bodies: List[RigidBody], collisions: Collisions):
        quadtree = QuadTree(self.box)
        for body in bodies:
            if body.collision:
                quadtree.add(body)

        quadtree.find_all_intersections(collisions)

    def find_all_intersections_basic(self, bodies: List[RigidBody], collisions: Collisions):
        for i in range(0, len(bodies)):
            a = bodies[i]
            if not a.collision:
                continue

            for j in range(0, i):
                b = bodies[j]
                if not b.collision:
                    continue

                if a.aabb.intersects(b.aabb):
                    collisions.append((a, b))
