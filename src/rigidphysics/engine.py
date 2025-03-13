"""Module providing the rigid physics engine.

NB some of this code is related to the need to render 3D graphics, which
complicates the drawing process. This is out of scope for the Tripos, but
students should understand the physics engine and how it interacts with
the rigid bodies.
"""

from collections import deque
from colorsys import hls_to_rgb
from typing import List, Mapping, NamedTuple, Set, Tuple
from pyglet.math import Vec3

import numpy as np

from .collisions import collide
from .contacts import find_contact_points
from .config import PrimitiveKind, RigidBodyConfig, SimulationConfig
from .detection import Detection
from .physics import Physics
from .primitives import Primitive
from .bodies import RigidBody, create_body


PrimitiveSpan = NamedTuple("PrimitiveSpan", [("name", str),
                                             ("primitive", Primitive),
                                             ("start", int),
                                             ("end", int),
                                             ("render", bool)])


class QueueState:
    """Represents the state of a primitive instance queue.

    Description:
        The engine can only display a fixed maximum number of instances.
        We want to be able to add and remove instances dynamically, so
        we use a queue to manage the indices of the instances that are
        currently in use. This class represents the state of that queue,
        and ensures that if a new shape is added such that the queue is
        full, the oldest shape is removed.

        NB this is out of scope for the Tripos, but the allocation algorithm
        may be of interest.
    """

    def __init__(self, start: int, capacity: int):
        self.allocated = deque()
        self.start = start
        self.capacity = capacity
        self.count = 0

    def allocate(self) -> int:
        """Return an instance index for use."""
        if self.count == self.capacity:
            # remove the oldest
            index = self.allocated.popleft()
        else:
            # allocated a new index
            index = self.start + self.count
            self.count += 1

        self.allocated.append(index)
        return index

    def free(self, index: int) -> int:
        """Free an instance index.

        Description:
            Because of how instances are rendered on the GPU the instances
            must be allocated in a block to avoid junk primitives. As such,
            when a primitive is freed, we need to copy the last primitive
            into the freed slot to maintain the block.

        Returns:
            int: The index that was copied back.
        """
        assert self.count == len(self.allocated)
        copy_back = self.start + self.count - 1
        for _ in range(self.count):
            i = self.allocated.popleft()
            if i == index:
                continue

            if i == copy_back:
                self.allocated.append(index)
            else:
                self.allocated.append(i)

        self.count -= 1
        return copy_back

    def is_full(self) -> bool:
        """Return whether the queue is full."""
        return self.count == self.capacity

    def clear(self):
        """Clear the queue."""
        self.allocated.clear()
        self.count = 0


class PhysicsEngine:
    """Class representing the rigid body physics engine."""

    # position + rotation + scale + color + alpha
    ELEMENTS_PER_INSTANCE = 3 + 4 + 3 + 3 + 1

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.gravity = np.array(config.gravity, np.float64)
        self.show_contacts = config.show_contacts
        self.contacts_state = None
        self.contacts = []
        self.contact_buffer = np.empty((25, 3), np.float64)
        self.physics = Physics(config.mode, config.restitution,
                               config.static_friction, config.dynamic_friction)
        self.create_spans()
        self.create_instance_buffer()
        self.num_bodies = 0
        self.bodies: Mapping[int, RigidBody] = {}
        self.changed: Set[int] = set()
        self.invisible: Set[int] = set()
        self.colliders: List[RigidBody] = []
        self.phantoms: Mapping[int, int] = {}
        self.num_detections = 0
        self.generate_bodies()

    def create_spans(self) -> Tuple[int, RigidBodyConfig, RigidBodyConfig]:
        """Out of scope."""
        self.primitives: Mapping[str, PrimitiveSpan] = {}
        self.primitive_queues: Mapping[str, QueueState] = {}
        start = 0
        self.contacts_config = None
        for body_config in self.config.bodies:
            for kind in body_config.kinds:
                name = body_config.name + "_" + str(kind)
                prim = Primitive.create(kind)
                end = start + body_config.max_count
                self.primitives[name] = PrimitiveSpan(name, prim, start, end, body_config.is_visible)
                self.primitive_queues[name] = QueueState(start, body_config.max_count)
                start = end
                if body_config.name == "contacts":
                    self.contacts_config = body_config
                    self.contacts_state = self.primitive_queues[name]

        self.detection = Detection(self.config.detection, start,
                                   self.config.spatial_cell_size)
        self.num_instances = start
        self.invisible_index = start

    def create_instance_buffer(self):
        """Out of scope."""
        self.instance_buffer = np.zeros((self.num_instances * PhysicsEngine.ELEMENTS_PER_INSTANCE),
                                        dtype=np.float32)
        start = 0
        end = self.num_instances * 3
        self.positions = self.instance_buffer[start:end].reshape(-1, 3)
        start = end
        end += self.num_instances * 4
        self.rotations = self.instance_buffer[start:end].reshape(-1, 4)
        start = end
        end += self.num_instances * 3
        self.scales = self.instance_buffer[start:end].reshape(-1, 3)
        start = end
        end += self.num_instances * 3
        self.colors = self.instance_buffer[start:end].reshape(-1, 3)
        start = end
        end += self.num_instances
        self.alpha = self.instance_buffer[start:end]

    def generate_bodies(self):
        """Out of scope."""
        if self.config.bounds.enabled:
            self.add_bounds()

        if self.contacts_config is not None:
            # add all the contacts in advance
            for _ in range(self.contacts_config.max_count):
                contact = create_body(self.contacts_config)
                hsv = self.contacts_config.color.sample_vec()
                rgb = hls_to_rgb(*hsv)
                self.add_body(contact, rgb)

            # we will manually manage this buffer
            self.contacts_state.count = 0

        for body_config in self.config.bodies:
            if "button" in body_config.name:
                # button bodies are created on demand
                continue

            for _ in range(body_config.max_count):
                body = create_body(body_config)
                hls = body_config.color.sample_vec()
                rgb = hls_to_rgb(*hls)
                self.add_body(body, rgb)
        
        self.detection.allocate_buffers()

    def add_bounds(self):
        """Out of scope."""
        scale = self.config.bounds.scale
        bounds_config = RigidBodyConfig.bounds(scale)

        half = scale / 2
        left = create_body(bounds_config)
        left.move_to([-half, 0, 0])
        left.rotate([0, 0, -np.pi/2])
        self.add_body(left, None)

        right = create_body(bounds_config)
        right.move_to([half, 0, 0])
        right.rotate([0, 0, -np.pi/2])
        self.add_body(right, None)

        top = create_body(bounds_config)
        top.move_to([0, half, 0])
        self.add_body(top, None)

        bottom = create_body(bounds_config)
        bottom.move_to([0, -half, 0])
        self.add_body(bottom, None)

        front = create_body(bounds_config)
        front.move_to([0, 0, half])
        front.rotate([np.pi/2, 0, 0])
        self.add_body(front, None)

        back = create_body(bounds_config)
        back.move_to([0, 0, -half])
        back.rotate([np.pi/2, 0, 0])
        self.add_body(back, None)

    def add_body(self, body: RigidBody, color: Vec3 = None):
        """Adds a body to the engine."""
        if color is None:
            body.index = self.invisible_index
            self.invisible_index += 1
            self.invisible.add(body.index)
        else:
            queue_state = self.primitive_queues[body.name]
            i = queue_state.allocate()
            body.index = i
            self.positions[i] = body.position
            self.rotations[i] = body.rotation
            self.scales[i] = body.size.astype(np.float32)
            self.colors[i] = color
            self.alpha[i] = 0.5 if body.physics else 1

        self.bodies[body.index] = body
        self.num_bodies = len(self.bodies)
        self.keys = list(self.bodies.keys())
        if body.collision:
            self.colliders.append(body)
            if not body.physics:
                self.detection.add_static_collider(body)

    def remove_body(self, body: RigidBody):
        """Removes a body from the engine."""
        if body.index in self.invisible:
            del self.bodies[body.index]
            return

        queue_state = self.primitive_queues[body.name]
        i = body.index
        j = queue_state.free(body.index)
        if i != j:
            self.bodies[i] = self.bodies[j]
            self.positions[i] = self.positions[j]
            self.rotations[i] = self.rotations[j]
            self.scales[i] = self.scales[j]
            self.colors[i] = self.colors[j]
            self.alpha[i] = self.alpha[j]
            self.bodies[i].index = i

        del self.bodies[j]
        self.num_bodies -= 1
        self.keys = list(self.bodies.keys())
        if body.collision:
            self.colliders.remove(body)

    def remove_all(self, kind: PrimitiveKind):
        """Removes all bodies of a given kind."""
        info = self.primitives[kind]
        self.primitive_queues[kind].clear()
        for i in range(info.start, info.end):
            if i in self.bodies:
                del self.bodies[i]

        self.num_bodies = len(self.bodies)
        self.keys = list(self.bodies.keys())

    def update_buffer(self):
        """Out of scope."""
        for i in self.changed:
            if i not in self.bodies or i >= self.num_instances:
                continue

            body = self.bodies[i]
            self.positions[i] = body.position.astype(np.float32)
            self.rotations[i] = body.rotation.astype(np.float32)

    def step(self, dt: float):
        """Step the simulation forward."""
        if self.show_contacts:
            self.contacts.clear()

        self.changed.clear()
        self.step_bodies(dt)
        self.broad_phase()
        self.narrow_phase(self.show_contacts)

        if self.show_contacts:
            # debug code for showing contacts
            if self.contacts:
                contacts = np.concatenate(self.contacts)
                contacts = np.unique(contacts, axis=0)
                count = len(contacts)
                if count > self.contacts_state.capacity:
                    count = self.contacts_state.capacity
                    contacts = contacts[:count]

                start = self.contacts_state.start
                end = start + count
                self.contacts_state.count = count
                self.positions[start:end] = contacts
            else:
                self.contacts_state.count = 0

        self.update_buffer()

    def step_bodies(self, dt: float):
        """Step the bodies forward in time."""
        gravity = self.gravity
        to_remove: List[RigidBody] = []
        min_bound = -2 * self.config.scale
        max_bound = 2 * self.config.scale
        for body in self.colliders:
            if not body.physics:
                # static body
                continue

            body.step(dt, gravity)
            pos = body.position
            if (pos[1] < min_bound or pos[0] < min_bound or pos[2] < min_bound or
                    pos[2] > max_bound or pos[1] > max_bound or pos[1] > max_bound):
                to_remove.append(body)
            else:
                self.changed.add(body.index)

        for body in to_remove:
            self.remove_body(body)

    def broad_phase(self):
        self.num_detections = self.detection.detect_collisions(self.colliders)

    def narrow_phase(self, add_contacts=False):
        for i in range(self.num_detections):
            a_idx, b_idx = self.detection.pairs[i]
            a = self.colliders[a_idx]
            b = self.colliders[b_idx]

            collision = collide(a, b)
            if collision is None:
                continue

            if self.physics.needs_contacts:
                num_contacts = find_contact_points(a, b, collision, self.contact_buffer)
                contacts = self.contact_buffer[:num_contacts]
            else:
                contacts = None

            self.physics.resolve_collision(a, b, collision, contacts)

            if a.physics:
                self.changed.add(a.index)

            if b.physics:
                self.changed.add(b.index)

            if not add_contacts:
                continue

            self.contacts.append(contacts.copy())
