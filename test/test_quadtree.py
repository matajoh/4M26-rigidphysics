

import random
from typing import List, NamedTuple, Tuple

import pytest
from rigidphysics.flat.bodies import AABB
from rigidphysics.flat.quadtree import Node, QuadTree


class MockBody(NamedTuple("MockBody", [("aabb", AABB), ("index", int)])):
    @staticmethod
    def create_random(index: int) -> "MockBody":
        left = random.uniform(0, 1)
        top = random.uniform(0, 1)
        width = min(1 - left, random.uniform(0, 0.01))
        height = min(1 - top, random.uniform(0, 0.01))
        return MockBody(AABB(left, top, left + width, top + height), index)


def generate_random_bodies(n: int) -> list[MockBody]:
    return [MockBody.create_random(i) for i in range(n)]


def query(box: AABB, bodies: List[MockBody], removed: List[bool]):
    intersections = []
    for b in bodies:
        if not removed or not removed[b.index]:
            if box.intersects(b.aabb):
                intersections.append(b)

    return intersections


def find_all_intersections(bodies: List[MockBody], removed: List[bool]):
    intersections = []
    num_bodies = len(bodies)
    for i in range(num_bodies):
        if not removed or not removed[i]:
            for j in range(0, i):
                if not removed or not removed[j]:
                    if bodies[i].aabb.intersects(bodies[j].aabb):
                        intersections.append((bodies[i], bodies[j]))

    return intersections


def check_intersection_bodies(bodies1: List[MockBody], bodies2: List[MockBody]):
    if len(bodies1) != len(bodies2):
        return False

    bodies1 = sorted([b1.index for b1 in bodies1])
    bodies2 = sorted([b2.index for b2 in bodies2])
    return all(i1 == i2 for i1, i2 in zip(bodies1, bodies2))


def check_intersections(intersections1: List[Tuple[MockBody, MockBody]],
                        intersections2: List[Tuple[MockBody, MockBody]]):
    if len(intersections1) != len(intersections2):
        return False

    intersections1 = [tuple(sorted([b1.index, b2.index]))
                      for b1, b2 in intersections1]
    intersections2 = [tuple(sorted([b1.index, b2.index]))
                      for b1, b2 in intersections2]
    intersections1.sort()
    intersections2.sort()
    return all(i1 == i2 for i1, i2 in zip(intersections1, intersections2))


def find(node: Node, box: AABB, value: MockBody):
    def find_(node: Node, box: AABB, value: MockBody):
        for body in node.values:
            if body.index == value.index:
                return box

        if node.is_leaf:
            return False

        for i, child in enumerate(node):
            child_box = find_(child, node.compute_box(i), value)
            if child_box:
                return child_box

    return find_(node, box, value)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_add_and_query(n: int):
    box = AABB(0, 0, 1, 1)
    bodies = generate_random_bodies(n)
    quadtree = QuadTree(box)
    for body in bodies:
        quadtree.add(body)

    intersections1 = []
    for body in bodies:
        intersections1.append(quadtree.query(body.aabb))

    intersections2 = []
    for body in bodies:
        intersections2.append(query(body.aabb, bodies, []))

    assert all(check_intersection_bodies(i1, i2)
               for i1, i2 in zip(intersections1, intersections2))


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_add_and_find_all_intersections(n: int):
    box = AABB(0, 0, 1, 1)
    bodies = generate_random_bodies(n)
    quadtree = QuadTree(box)
    for body in bodies:
        quadtree.add(body)

    intersections1 = []
    quadtree.find_all_intersections(intersections1)
    intersections2 = find_all_intersections(bodies, [])
    assert check_intersections(intersections1, intersections2)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_add_remove_and_query(n: int):
    box = AABB(0, 0, 1, 1)
    bodies = generate_random_bodies(n)
    quadtree = QuadTree(box)
    for body in bodies:
        quadtree.add(body)
        box = find(quadtree.root, quadtree.box, body)
        assert box

    removed = [random.random() < 0.5 for _ in range(n)]
    for i in range(n):
        if removed[i]:
            quadtree.remove(bodies[i])

    intersections1 = []
    for i, body in enumerate(bodies):
        if not removed[i]:
            intersections1.append(quadtree.query(body.aabb))

    intersections2 = []
    for i, body in enumerate(bodies):
        if not removed[i]:
            intersections2.append(query(body.aabb, bodies, removed))

    assert all(check_intersection_bodies(i1, i2)
               for i1, i2 in zip(intersections1, intersections2))


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_add_remove_and_find_all_intersections(n: int):
    box = AABB(0, 0, 1, 1)
    bodies = generate_random_bodies(n)
    quadtree = QuadTree(box)
    for body in bodies:
        quadtree.add(body)

    removed = [random.random() < 0.5 for _ in range(n)]
    for i in range(n):
        if removed[i]:
            quadtree.remove(bodies[i])

    intersections1 = []
    quadtree.find_all_intersections(intersections1)
    intersections2 = find_all_intersections(bodies, removed)
    assert check_intersections(intersections1, intersections2)


if __name__ == "__main__":
    test_add_and_query(100)
    test_add_and_find_all_intersections(100)
    test_add_remove_and_query(100)
    test_add_remove_and_find_all_intersections(100)
