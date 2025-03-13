

from typing import List, Set, Tuple
import numpy as np
from random import random

from rigidphysics.config import DetectionKind, RigidBodyConfig
from rigidphysics.detection import Detection
from rigidphysics.bodies import RigidBody, create_body


floor, ledge0, ledge1 = RigidBodyConfig.defaults(30)


cube_config = RigidBodyConfig.from_dict({
    "name": "cuboid",
    "position": {
        "min": [-15, -15, -15],
        "max": [15, 15, 15]
    },
    "size": {
        "min": 2,
        "max": 3
    },
    "velocity": {
        "min": [-5, -5, -5],
        "max": [5, 5, 5]
    },
    "angular_velocity": {
        "min": [-2, -2, -2],
        "max": [2, 2, 2]
    },
    "rotation": 0
})


sphere_config = RigidBodyConfig.from_dict({
    "name": "sphere",
    "position": {
        "min": [-15, -15, -15],
        "max": [15, 15, 15]
    },
    "size": {
        "min": 1,
        "max": 1.5
    },
    "velocity": {
        "min": [-5, -5, -5],
        "max": [5, 5, 5]
    },
    "angular_velocity": {
        "min": [-2, -2, -2],
        "max": [2, 2, 2]
    },
    "rotation": 0
})


def generate_colliders(count, cube_prob) -> List[RigidBody]:
    static_colliders = [create_body(floor), create_body(ledge0), create_body(ledge1)]
    static_colliders[0].index = 0
    static_colliders[1].index = 1
    static_colliders[2].index = 2
    colliders = []
    for i in range(count):
        body = create_body(cube_config if random() < cube_prob else sphere_config)
        body.index = i + 3
        colliders.append(body)

    return static_colliders, colliders


def expected_detections(colliders: List[RigidBody]) -> Set[Tuple[int, int]]:
    detections = set()
    for i, a in enumerate(colliders):
        a_min, a_max = a.aabb
        for j, b in enumerate(colliders[i+1:], start=i+1):
            b_min, b_max = b.aabb
            if np.any(a_max < b_min) or np.any(b_max < a_min):
                continue

            detections.add((i, j))

    return detections


def test_basic():
    num_dynamic = 200
    static_colliders, colliders = generate_colliders(num_dynamic, 0.5)
    num_static = len(static_colliders)
    spacing = 3
    detection = Detection(DetectionKind.BASIC, num_dynamic, spacing)
    for i in range(num_static):
        detection.add_static_collider(static_colliders[i])

    detection.allocate_buffers()

    colliders = static_colliders + colliders
    num_collisions = detection.detect_collisions(colliders)
    expected = expected_detections(colliders)
    actual = set()
    for i in range(num_collisions):
        a, b = detection.pairs[i]
        if a < b:
            actual.add((a, b))
        else:
            actual.add((b, a))

    assert actual == expected


def test_spatial_hashing():
    num_dynamic = 200
    static_colliders, colliders = generate_colliders(num_dynamic, 0.5)
    num_static = len(static_colliders)
    spacing = 3
    detection = Detection(DetectionKind.SPATIAL_HASHING, num_dynamic, spacing)
    for i in range(num_static):
        detection.add_static_collider(static_colliders[i])

    detection.allocate_buffers()

    colliders = static_colliders + colliders
    num_collisions = detection.detect_collisions(colliders)
    expected = expected_detections(colliders)
    actual = set()
    for i in range(num_collisions):
        a, b = detection.pairs[i]
        a, b = int(a), int(b)
        if a < b:
            actual.add((a, b))
        else:
            actual.add((b, a))

    assert actual == expected, (len(actual), len(expected))


if __name__ == "__main__":
    test_spatial_hashing()
