"""Module providing a convex hull of a set of points.

Description:
    Note that this module is out of scope for the Tripos. However, students may
    find it interesting to see how a Graham scan is implemented for computing the 
    convex hull of a set of points. The code is based on the implementation in
    https://en.wikipedia.org/wiki/Graham_scan.
"""

from typing import List, NamedTuple
from pygame import Vector2


def area_sign(v0: Vector2, v1: Vector2, v2: Vector2) -> float:
    area2 = (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y)
    if area2 > 1e-5:
        return 1

    if area2 < -1e-5:
        return -1

    return 0


class Point:
    def __init__(self, vnum: int, v: Vector2):
        self.vnum = vnum
        self.v = v
        self.delete = False


class PointKey(NamedTuple("PointKey", [("p", Point), ("v0", Vector2)])):
    def compare(self, other: "PointKey") -> bool:
        pi = self.p
        pj = other.p
        a = area_sign(self.v0, pi.v, pj.v)
        if a > 0:
            return -1

        if a < 0:
            return 1

        # colinear with v0
        x = abs(pi.v.x - self.v0.x) - abs(pj.v.x - self.v0.x)
        y = abs(pi.v.y - self.v0.y) - abs(pj.v.y - self.v0.y)
        if x < 0 or y < 0:
            pi.delete = True
            return -1

        if x > 0 or y > 0:
            pj.delete = True
            return 1

        # pi and pj are coincident
        if pi.vnum > pj.vnum:
            pj.delete = True
        else:
            pi.delete = True

        return 0

    def __lt__(self, other: "PointKey") -> bool:
        return self.compare(other) < 0

    def __le__(self, other: "PointKey") -> bool:
        return self.compare(other) <= 0

    def __gt__(self, other: "PointKey") -> bool:
        return self.compare(other) > 0

    def __ge__(self, other: "PointKey") -> bool:
        return self.compare(other) >= 0

    def __eq__(self, other: "PointKey") -> bool:
        return self.compare(other) == 0

    def __ne__(self, other: "PointKey") -> bool:
        return self.compare(other) != 0


def compute_convex_hull(vertices: List[Vector2]) -> List[Vector2]:
    v0 = vertices[0]
    for i in range(1, len(vertices)):
        v = vertices[i]
        if v.y > v0.y:
            v0 = v
        elif v.y == v0.y and v.x > v0.x:
            v0 = v

    vertices.remove(v0)

    points = [Point(i + 1, v) for i, v in enumerate(vertices)]
    points.sort(key=lambda p: PointKey(p, v0))
    points = [p for p in points
              if not p.delete]
    points.insert(0, Point(0, v0))
    # Graham scan (https://en.wikipedia.org/wiki/Graham_scan)
    stack = [points[0], points[1]]
    i = 2
    n = len(points)
    while i < n:
        p1, p2 = stack[-2], stack[-1]
        if area_sign(p1.v, p2.v, points[i].v) > 0:
            # p[i] is to the left of ab -> clockwise turn
            stack.append(points[i])
            i += 1
        else:
            stack.pop()

    return [p.v for p in stack]
