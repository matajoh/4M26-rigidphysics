import time
from typing import List, Tuple

import numpy as np
import random
from tqdm import tqdm

from rigidphysics.collisions import collide
from rigidphysics.config import DetectionKind, PhysicsMode, RigidBodyConfig
from rigidphysics.detection import Detection
from rigidphysics.physics import Physics
from rigidphysics.contacts import find_contact_points
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


def generate_colliders(count, cube_prob, num_static) -> Tuple[List[RigidBody], List[RigidBody]]:
    colliders = []
    for i in range(count):
        body = create_body(cube_config if random.random() < cube_prob else sphere_config)
        body.index = num_static + i
        colliders.append(body)

    return colliders


def time_collision(count=100, num_dynamic=1000, cube_prob=0.5,
                   detection_kind=DetectionKind.BASIC, physics_mode=PhysicsMode.BASIC):
    contacts = np.empty((25, 3), np.float64)

    # warmup to compile
    print("compiling...")
    static_colliders = [create_body(floor), create_body(ledge0), create_body(ledge1)]
    static_colliders[0].index = 0
    static_colliders[1].index = 1
    static_colliders[2].index = 2
    num_static = len(static_colliders)
    colliders = generate_colliders(num_dynamic, cube_prob, num_static)
    detection = Detection(detection_kind, num_dynamic, 3)
    physics = Physics(physics_mode, 0.5, 0.5, 0.5)

    for b in static_colliders:
        detection.add_static_collider(b)

    detection.allocate_buffers()

    colliders = static_colliders + colliders
    num_collisions = detection.detect_collisions(colliders)

    for i in range(num_collisions):
        a_idx, b_idx = detection.pairs[i]
        a = colliders[a_idx]
        b = colliders[b_idx]

        collision = collide(a, b)
        if collision is None:
            continue

        num_contacts = find_contact_points(a, b, collision, contacts)
        if num_contacts == 0:
            continue

        physics.resolve_collision(a, b, collision, contacts[:num_contacts])

    print("starting")
    detect_elapsed = 0
    collide_elapsed = 0
    contacts_elapsed = 0
    resolve_elapsed = 0
    contacts_sum = 0
    detect_count = 0
    collide_count = 0
    for _ in tqdm(range(count), "timing collisions"):
        colliders[num_static:] = generate_colliders(num_dynamic, cube_prob, num_static)

        start = time.perf_counter()
        num_collisions = detection.detect_collisions(colliders)
        detect_elapsed += time.perf_counter() - start
        detect_count += num_collisions

        for i in range(num_collisions):
            a_idx, b_idx = detection.pairs[i]
            a = colliders[a_idx]
            b = colliders[b_idx]

            start = time.perf_counter()
            collision = collide(a, b)
            collide_elapsed += time.perf_counter() - start

            if collision is None:
                continue

            collide_count += 1
            start = time.perf_counter()
            num_contacts = find_contact_points(a, b, collision, contacts)
            contacts_elapsed += time.perf_counter() - start

            if num_contacts == 0:
                continue

            contacts_sum += num_contacts

            start = time.perf_counter()
            physics.resolve_collision(a, b, collision, contacts[:num_contacts])
            resolve_elapsed += time.perf_counter() - start

    total_elapsed = detect_elapsed + collide_elapsed + contacts_elapsed + resolve_elapsed
    time_per_step = 1000 * total_elapsed / count
    time_per_detect = 1000 * detect_elapsed / count
    time_per_collide = 1000 * collide_elapsed / count
    time_per_contacts = 1000 * contacts_elapsed / count
    time_per_resolve = 1000 * resolve_elapsed / count
    print("# detections:", detect_count)
    print("# collisions:", collide_count)
    print(f"# contacts: {contacts_sum / collide_count:.2f}")
    print(f"Time per step: {time_per_step:.4f}ms")
    print(f"Time per detect: {time_per_detect:.4f}ms")
    print(f"Time per collide: {time_per_collide:.4f}ms")
    print(f"Time per contacts: {time_per_contacts:.4f}ms")
    print(f"Time per resolve: {time_per_resolve:.4f}ms")


if __name__ == "__main__":
    random.seed(20080524)
    time_collision(10, 9192, 0.8, DetectionKind.BASIC, PhysicsMode.FRICTION)
    time_collision(10, 9192, 0.8, DetectionKind.SPATIAL_HASHING, PhysicsMode.FRICTION)
