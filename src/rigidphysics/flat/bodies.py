"""Module providing basic circle and polygon bodies.

Revision activity: try adding a pentagon!
"""

import math

from typing import List, NamedTuple, Tuple, Union

from pygame import Vector2
import pygame


Color = Tuple[int, int, int]


class AABB(NamedTuple("AABB", [("left", float), ("top", float), ("right", float), ("bottom", float)])):
    def disjoint(self, other: "AABB") -> bool:
        """Check if this AABB is disjoint from another AABB.

        Description:
            Note how this test is designed such that it never needs to
            do more than the minimum number of comparisons.
        """
        return (self.left > other.right or
                self.right < other.left or
                self.top > other.bottom or
                self.bottom < other.top)

    def contains(self, other: "AABB") -> bool:
        """Check if this AABB contains another AABB.

        Description:
            This test is designed to be as efficient as possible.
        """
        return not (self.left > other.left or
                    self.right < other.right or
                    self.top > other.top or
                    self.bottom < other.bottom)

    def intersects(self, other: "AABB") -> bool:
        """Check if this AABB intersects with another AABB.

        Description:
            Using the negation of disjoint means that we can avoid
            unnecessary tests. This is an example of De Morgan's law.
        """
        return not self.disjoint(other)

    @property
    def top_left(self) -> Vector2:
        """Get the top left corner of the AABB."""
        return Vector2(self.left, self.top)

    @property
    def center(self) -> Vector2:
        """Get the center of the AABB."""
        return Vector2((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    @property
    def size(self) -> Vector2:
        """Get the size of the AABB."""
        return Vector2(self.right - self.left, self.bottom - self.top)

    @staticmethod
    def create(top_left: Vector2, size: Vector2) -> "AABB":
        return AABB(top_left.x, top_left.y, top_left.x + size.x, top_left.y + size.y)


class Circle:
    """A circle body."""

    def __init__(self,
                 radius: float,
                 color: Color,
                 linear_velocity: Vector2,
                 angular_velocity: float,
                 mass: float,
                 inv_mass: float,
                 inertia: float,
                 inv_inertia: float):
        self.position = Vector2(0, 0)
        self.angle = 0
        self.radius = radius
        self.color = color
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.mass = mass
        self.inv_mass = inv_mass
        self.inertia = inertia
        self.inv_inertia = inv_inertia
        self.size = radius * 2
        self.aabb_ = AABB(0, 0, 0, 0)
        self.update_needed_ = True

    def draw(self, screen: pygame.Surface, center: Vector2, scale: float):
        """NB not in scope for Tripos."""
        pos = (self.position + center) * scale
        radius = self.radius * scale
        cos = math.cos(self.angle)
        sin = math.sin(self.angle)
        p = Vector2(cos, sin) * radius * 0.9
        pygame.draw.circle(screen, self.color, pos, radius)
        pygame.draw.circle(screen, "black", pos, radius, 4)
        pygame.draw.line(screen, "black", pos, pos + p, 4)

    def step(self, dt: float, gravity: Vector2):
        self.linear_velocity += gravity * dt
        self.position += self.linear_velocity * dt
        self.angle += self.angular_velocity * dt
        self.update_needed_ = True

    def move_to(self, pos: Vector2) -> "Circle":
        self.position = pos
        self.update_needed_ = True
        return self

    def move(self, delta: Vector2) -> "Circle":
        self.position += delta
        self.update_needed_ = True
        return self

    def to_dict(self):
        return {"kind": "circle",
                "position": (self.position.x, self.position.y),
                "angle": self.angle,
                "radius": self.radius,
                "color": self.color}

    @property
    def aabb(self) -> AABB:
        """Get the axis-aligned bounding box of the circle.

        Description:
            Note how the bounding box is only updated when necessary.
        """
        if self.update_needed_:
            self.aabb_ = AABB(self.position.x - self.radius, self.position.y - self.radius,
                              self.position.x + self.radius, self.position.y + self.radius)
            self.update_needed_ = False

        return self.aabb_

    @staticmethod
    def create(radius: float, density: float, color: Color):
        mass = radius**2 * math.pi * density
        inv_mass = 1 / mass
        # The moment of inertia of a circle is 0.5 * mass * radius^2
        # when interpreted as a "thin solid disk"
        # see: https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        inertia = 0.5 * mass * radius * radius
        inv_inertia = 1 / inertia
        return Circle(radius, color,
                      Vector2(0, 0), 0,
                      mass, inv_mass,
                      inertia, inv_inertia)


class Polygon:
    """A polygon body.

    Description:
        Polygons can be either static or dynamic. As such, the
        constructor provides default values for static polygons
        which will ensure they interact properly with dynamic objects.
    """

    def __init__(self,
                 vertices: List[Vector2],
                 normals: List[Vector2],
                 color: Color,
                 linear_velocity: Vector2 = None,
                 angular_velocity: float = None,
                 mass: float = float("inf"),
                 inv_mass: float = 0,
                 inertia: float = float("inf"),
                 inv_inertia: float = 0):
        self.position = Vector2(0, 0)
        self.angle = 0
        self.vertices = vertices
        self.normals = normals
        self.radius = max(v.length() for v in vertices)
        self.color = color
        self.mass = mass
        self.inv_mass = inv_mass
        self.inertia = inertia
        self.inv_inertia = inv_inertia
        if linear_velocity is not None:
            self.linear_velocity = linear_velocity
            self.angular_velocity = angular_velocity

        self.aabb_ = AABB(0, 0, 0, 0)
        self.transformed_vertices_ = self.vertices.copy()
        self.transformed_normals_ = self.normals.copy()
        self.update_needed_ = True

    def draw(self, screen: pygame.Surface, center: Vector2, scale: float):
        """NB not in scope for Tripos."""
        vertices = [(v + center) * scale for v in self.transformed_vertices]
        pygame.draw.polygon(screen, self.color, vertices)
        pygame.draw.polygon(screen, "black", vertices, 4)

    def step(self, dt: float, gravity: Vector2):
        self.linear_velocity += gravity * dt
        self.position += self.linear_velocity * dt
        self.angle += self.angular_velocity * dt
        self.update_needed_ = True

    def move_to(self, pos: Vector2) -> "Polygon":
        self.position = pos
        self.update_needed_ = True
        return self

    def move(self, delta: Vector2) -> "Polygon":
        self.position += delta
        self.update_needed_ = True
        return self

    def rotate_to(self, angle: float) -> "Polygon":
        self.angle = angle
        self.update_needed_ = True
        return self

    def update_transform(self):
        """Update the transformed vertices and normals.

        Description:
            We want to avoid updating the transformed vertices and normals
            unless absolutely necessary. This is an example of lazy
            evaluation using a "dirty" bit, in this case the update_needed_
            flag.
        """
        if not self.update_needed_:
            return

        self.update_needed_ = False
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        for i, n in enumerate(self.normals):
            self.transformed_normals_[i] = Vector2(n.x * cos_angle - n.y * sin_angle,
                                                   n.x * sin_angle + n.y * cos_angle)

        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        for i, v in enumerate(self.vertices):
            tv = Vector2(v.x * cos_angle - v.y * sin_angle,
                         v.x * sin_angle + v.y * cos_angle)
            tv += self.position
            min_x = min(min_x, tv.x)
            min_y = min(min_y, tv.y)
            max_x = max(max_x, tv.x)
            max_y = max(max_y, tv.y)
            self.transformed_vertices_[i] = tv

        self.aabb_ = AABB(min_x, min_y, max_x, max_y)

    def to_dict(self):
        return {"kind": "polygon",
                "position": (self.position.x, self.position.y),
                "angle": self.angle,
                "vertices": [(v.x, v.y) for v in self.vertices],
                "color": self.color}

    @property
    def aabb(self) -> AABB:
        self.update_transform()
        return self.aabb_

    @property
    def transformed_vertices(self) -> List[Vector2]:
        self.update_transform()
        return self.transformed_vertices_

    @property
    def transformed_normals(self) -> List[Vector2]:
        self.update_transform()
        return self.transformed_normals_

    @staticmethod
    def create_rectangle(width: float, height: float, density: float,
                         color: Color, is_static=False):
        vertices = [Vector2(-width / 2, -height / 2),
                    Vector2(width / 2, -height / 2),
                    Vector2(width / 2, height / 2),
                    Vector2(-width / 2, height / 2)]
        normals = [Vector2(0, 1), Vector2(1, 0)]
        if is_static:
            return Polygon(vertices, normals, color)

        mass = width * height * density
        inv_mass = 1 / mass
        # The moment of inertia of a rectangle is (1 / 12) * mass * (width^2 + height^2)
        # when interpreted as a "thin rectangular plate"
        # see: https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        inertia = (1 / 12) * mass * (width**2 + height**2)
        inv_inertia = 1 / inertia

        return Polygon(vertices, normals, color,
                       Vector2(0, 0), 0,
                       mass, inv_mass,
                       inertia, inv_inertia)

    @staticmethod
    def create_regular_polygon(num_sides: int, radius: float, density: float,
                               color: Color, is_static=False):
        angle = 2 * math.pi / num_sides
        vertices = [Vector2(radius * math.cos(i * angle), radius * math.sin(i * angle))
                    for i in range(num_sides)]
        normals = []
        if num_sides % 2 == 0:
            for i in range(num_sides // 2):
                n = (vertices[i] + vertices[i + 1]).normalize()
                normals.append(n)
        else:
            for i in range(num_sides):
                n = (vertices[i] + vertices[(i + 1) % num_sides]).normalize()
                normals.append(n)

        if is_static:
            return Polygon(Vector2(0, 0), 0, vertices, normals, color)

        apothem = radius * math.cos(angle / 2)
        perimeter = num_sides * (vertices[1] - vertices[0]).length()        
        mass = 0.5 * perimeter * apothem * density
        inv_mass = 1 / mass
        # The moment of inertia of a regular polygon is 1/2 m s^2 (1 - 2/3 sin^2(\pi / n))
        # see: https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        inertia = 0.5 * mass * radius**2 * (1 - 2 / 3 * math.sin(math.pi / num_sides)**2)
        inv_inertia = 1 / inertia

        return Polygon(vertices, normals, color,
                       Vector2(0, 0), 0,
                       mass, inv_mass,
                       inertia, inv_inertia)


RigidBody = Union[Circle, Polygon]
