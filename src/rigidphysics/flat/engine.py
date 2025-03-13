"""Module containing the physics engine."""

from typing import List, Set, Tuple

from pygame import Vector2
import pygame

from ..config import DetectionKind, PhysicsMode
from .bodies import AABB, RigidBody
from .collisions import detect_collision
from .contacts import find_contact_points
from .detection import Detection
from .physics import Physics


class PhysicsEngine:
    """Class representing the physics engine.

    Description:
        The physics engine is in charge of maintaining the list of 
        rigid bodies and causing them to interact with each other, while
        managing the state of the simulation.

    Args:
        width (float): The width (in pixels) of the simulation window
        height (float): The height (in pixels) of the simulation window
        mode (PhysicsMode): The physics mode to use
        detection_kind (DetectionKind): The collision detection algorithm to use
        show_contacts (bool): Whether to display contact points
        height_in_meters (float): The height of the simulation window in meters
    """

    def __init__(self, width: float, height: float,
                 mode: PhysicsMode, detection_kind: DetectionKind,
                 show_contacts: bool, height_in_meters=30):
        self.scale = height / height_in_meters
        self.width = width / self.scale
        self.height = height_in_meters
        self.physics = Physics(mode)
        self.bounds = AABB(-self.width / 2, -self.width / 2,
                           self.width / 2, self.height / 2)
        self.detection = Detection(detection_kind, AABB(-self.width, -self.height,
                                                        self.width, self.height))
        self.bodies: List[RigidBody] = []
        self.gravity = Vector2(0, 9.81)
        self.collisions: List[Tuple[RigidBody, RigidBody]] = []
        self.to_remove: List[RigidBody] = []
        self.center = Vector2(self.width / 2, self.height / 2)
        self.contacts: Set[Vector2] = set()
        self.show_contacts = show_contacts
        self.mode = mode
        # the systems are defined by which components they operate over
        self.systems = {
            "physics": ["position", "angle",
                        "linear_velocity", "angular_velocity",
                        "mass", "inertia"],
            "collision": ["aabb"],
            "render": ["draw"]
        }

    def remove_outside(self):
        """Removes bodies that are outside the bounds of the simulation."""
        self.to_remove.clear()
        for body in self.bodies:
            if body.aabb.disjoint(self.bounds):
                # if the body is outside the bounds of the simulation,
                # we can safely remove it
                self.to_remove.append(body)

        for body in self.to_remove:
            self.bodies.remove(body)

    def broad_phase(self):
        """Performs the broad phase of collision detection.

        Description:
            This phase focuses on determining quickly which pairs of bodies
            are potentially colliding. It does this by checking if the
            bounding boxes of the bodies intersect.
        """
        return self.detection.find_all_intersections(self.bodies, self.collisions)

    def narrow_phase(self):
        """Performs the narrow phase of collision detection.

        Description:
            This phase focuses on determining the exact nature of the
            collision between two bodies, and then resolving it by
            passing the information to the physics module.
        """
        for a, b in self.collisions:
            collision = detect_collision(a, b)
            if collision is None:
                # false positive
                continue

            c0, c1 = find_contact_points(a, b, collision)
            if self.show_contacts:
                self.contacts.add((c0.x, c0.y))
                if c1 is not None:
                    self.contacts.add((c1.x, c1.y))

            self.physics.resolve_collision(a, b, collision, c0, c1)

    def step(self, dt: float, num_iterations=20):
        """Advances the simulation by a time step.

        Args:
            dt (float): The time step to advance the simulation by
            num_iterations (int): The number of iterations to perform
                                  in each time step. More iterations results
                                  in more accurate physics, but is more
                                  computationally expensive.
        """
        # the main problem inherent in a posteriori physics simulation
        # is "tunneling" where objects pass through each other.
        # to mitigate this, we subdivide the time step into smaller
        # increments and resolve collisions at each step using impulses.
        dt /= num_iterations
        self.contacts.clear()
        for i in range(num_iterations):
            for body in self.bodies:
                if body.physics:
                    body.step(dt, self.gravity)

            self.remove_outside()

            self.collisions.clear()
            self.broad_phase()
            self.narrow_phase()

    def draw(self, screen: pygame.Surface):
        """Draws the simulation to the screen."""
        for body in self.bodies:
            if body.render:
                body.draw(screen, self.center, self.scale)

        for contact in self.contacts:
            pygame.draw.circle(screen, "yellow", (contact + self.center) * self.scale, 5)
            pygame.draw.circle(screen, "black", (contact + self.center) * self.scale, 5, 2)

    def add_body(self, body: RigidBody):
        """Adds a body to the simulation."""
        for system, components in self.systems.items():
            # we will store a boolean for each system indicating
            # whether the body has all the necessary components
            has_system = all(hasattr(body, component) for component in components)
            setattr(body, system, has_system)

        self.bodies.append(body)

    def remove_body(self, body: RigidBody):
        """Removes a body from the simulation."""
        self.bodies.remove(body)

    def to_world(self, pos: Vector2) -> Vector2:
        """Converts a position from screen coordinates to world coordinates.

        Args:
            pos (Vector2): The position to convert

        Returns:
            Vector2: The position in world coordinates
        """
        return pos / self.scale - self.center
