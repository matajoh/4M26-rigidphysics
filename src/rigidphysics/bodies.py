"""Module providing code for spheres and cuboids."""

import math
from random import choice
from typing import Union

import numpy as np
from numba import jit

from .config import PrimitiveKind, RigidBodyConfig
from .geometry import CUBE_NORMALS, CUBE_VERTICES
from .maths import (
    quaternion_compose,
    quaternion_from_rotvec,
    quaternion_identity,
    quaternion_rotate_batch,
    quaternion_to_matrix
)


@jit(cache=True)
def update_transform_sphere(rotation: np.ndarray,
                            rot_mat: np.ndarray,
                            inv_inertia: np.ndarray,
                            inv_inertia_0: np.ndarray,
                            aabb: np.ndarray,
                            position: np.ndarray,
                            radius: float):
    quaternion_to_matrix(rotation, rot_mat)
    inv_inertia[:] = rot_mat @ inv_inertia_0 @ rot_mat.T
    aabb[0] = position - radius
    aabb[1] = position + radius


@jit(cache=True)
def update_transform_cuboid(rotation: np.ndarray,
                            rot_mat: np.ndarray,
                            inv_inertia: np.ndarray,
                            inv_inertia_0: np.ndarray,
                            aabb: np.ndarray,
                            position: np.ndarray,
                            vertices: np.ndarray,
                            vertices_0: np.ndarray,
                            normals: np.ndarray,
                            normals_0: np.ndarray):
    quaternion_to_matrix(rotation, rot_mat)
    inv_inertia[:] = rot_mat @ inv_inertia_0 @ rot_mat.T

    quaternion_rotate_batch(rotation, vertices_0, vertices)
    vertices += position
    for i in range(3):
        aabb[0, i] = np.min(vertices[:, i])
        aabb[1, i] = np.max(vertices[:, i])

    quaternion_rotate_batch(rotation, normals_0, normals)


@jit(cache=True)
def step_(dt: float,
          linear_velocity: np.ndarray,
          gravity: np.ndarray,
          angular_velocity: np.ndarray,
          rv_quat: np.ndarray,
          position: np.ndarray,
          rotation: np.ndarray):
    linear_velocity += gravity * dt
    position += linear_velocity * dt
    quaternion_from_rotvec(angular_velocity * dt, rv_quat)
    quaternion_compose(rv_quat, rotation, rotation)


class Sphere:
    """Class representing a sphere primitive."""

    def __init__(self,
                 radius: float,
                 linear_velocity: np.ndarray,
                 angular_velocity: np.ndarray,
                 mass: float,
                 inv_mass: float,
                 inertia: np.ndarray,
                 inv_inertia: np.ndarray):
        self.kind = PrimitiveKind.SPHERE
        self.position = np.zeros(3, np.float64)
        self.rotation = quaternion_identity()
        self.radius = radius
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.mass = mass
        self.inv_mass = inv_mass
        self.inertia_0 = inertia
        self.inv_inertia_0 = inv_inertia
        self.size = np.array([radius * 2, radius * 2, radius * 2], np.float64)

        self.inv_inertia_ = np.empty_like(inv_inertia)
        self.rot_mat_ = np.empty((3, 3), np.float64)
        self.rv_quat_ = np.empty(4, np.float64)
        self.aabb_ = np.zeros((2, 3), np.float64)
        self.update_needed_ = True

    def step(self, dt: float, gravity: np.ndarray):
        step_(dt, self.linear_velocity, gravity, self.angular_velocity,
              self.rv_quat_, self.position, self.rotation)
        self.update_needed_ = True

    def move(self, amount: np.ndarray):
        self.position += amount
        self.update_needed_ = True

    def move_to(self, position: np.ndarray):
        self.position[:] = position
        self.update_needed_ = True

    def rotate(self, rotvec: np.ndarray):
        rotvec = np.array(rotvec, np.float64)
        quaternion_from_rotvec(rotvec, self.rv_quat_)
        quaternion_compose(self.rotation, self.rv_quat_, self.rotation)
        self.update_needed_ = True

    def rotate_to(self, rotvec: np.ndarray):
        rotvec = np.array(rotvec, np.float64)
        quaternion_from_rotvec(rotvec, self.rotation)
        self.update_needed_ = True

    def update_transform(self):
        if self.update_needed_:
            self.update_needed_ = False
            update_transform_sphere(self.rotation, self.rot_mat_, self.inv_inertia_,
                                    self.inv_inertia_0, self.aabb_, self.position,
                                    self.radius)

    @property
    def aabb(self) -> np.ndarray:
        self.update_transform()
        return self.aabb_

    @property
    def inv_inertia(self) -> np.ndarray:
        self.update_transform()
        return self.inv_inertia_

    @staticmethod
    def create(radius: float, density: float) -> "Sphere":
        volume = 4 / 3 * math.pi * radius ** 3
        mass = density * volume
        inv_mass = 1 / mass
        # The moment of inertia of a sphere is 2/5 * mass * radius^2
        # https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        inertia = np.eye(3, dtype=np.float64) * (2 / 5) * mass * radius ** 2
        inv_inertia = np.linalg.inv(inertia)
        return Sphere(radius,
                      np.zeros(3, np.float64), np.zeros(3, np.float64),
                      mass, inv_mass, inertia, inv_inertia)


IFNTY_3x3 = np.full((3, 3), float("inf"), np.float64)
ZERO_3x3 = np.zeros((3, 3), np.float64)


class Cuboid:
    """Class representing a cuboid primitive."""

    def __init__(self,
                 size: np.ndarray,
                 vertices: np.ndarray,
                 normals: np.ndarray,
                 linear_velocity: np.ndarray = None,
                 angular_velocity: np.ndarray = None,
                 mass: float = float("inf"),
                 inv_mass: float = 0,
                 inertia: np.ndarray = IFNTY_3x3,
                 inv_inertia: float = ZERO_3x3):
        self.kind = PrimitiveKind.CUBOID
        self.position = np.zeros(3, np.float64)
        self.rotation = quaternion_identity()
        self.size = size
        self.vertices = vertices
        self.normals = normals
        self.radius = float(np.linalg.norm(vertices, axis=1).max(0))
        self.mass = mass
        self.inv_mass = inv_mass
        self.inertia_0 = inertia
        self.inv_inertia_0 = inv_inertia
        if linear_velocity is not None:
            self.linear_velocity = linear_velocity
            self.angular_velocity = angular_velocity

        self.rot_mat_ = np.empty((3, 3), np.float64)
        self.rv_quat_ = np.empty(4, np.float64)
        self.inv_inertia_ = np.empty_like(inv_inertia)
        self.aabb_ = np.empty((2, 3), np.float64)
        self.transformed_vertices_ = np.empty_like(vertices)
        self.transformed_normals_ = np.empty_like(normals)
        self.update_needed_ = True

    def step(self, dt: float, gravity: np.ndarray):
        step_(dt, self.linear_velocity, gravity, self.angular_velocity,
              self.rv_quat_, self.position, self.rotation)
        self.update_needed_ = True

    def move(self, amount: np.ndarray):
        self.position += amount
        self.update_needed_ = True

    def move_to(self, position: np.ndarray):
        self.position[:] = position
        self.update_needed_ = True

    def rotate(self, rotvec: np.ndarray):
        rotvec = np.array(rotvec, np.float64)
        quaternion_from_rotvec(rotvec, self.rv_quat_)
        quaternion_compose(self.rotation, self.rv_quat_, self.rotation)
        self.update_needed_ = True

    def rotate_to(self, rotvec: np.ndarray):
        rotvec = np.array(rotvec, np.float64)
        quaternion_from_rotvec(rotvec, self.rotation)
        self.update_needed_ = True

    @property
    def aabb(self) -> np.ndarray:
        self.update_transform()
        return self.aabb_

    @property
    def inv_inertia(self) -> np.ndarray:
        self.update_transform()
        return self.inv_inertia_

    @property
    def transformed_vertices(self) -> np.ndarray:
        self.update_transform()
        return self.transformed_vertices_

    @property
    def transformed_normals(self) -> np.ndarray:
        self.update_transform()
        return self.transformed_normals_

    def update_transform(self):
        if self.update_needed_:
            self.update_needed_ = False
            update_transform_cuboid(self.rotation, self.rot_mat_, self.inv_inertia_,
                                    self.inv_inertia_0, self.aabb_, self.position,
                                    self.transformed_vertices_, self.vertices,
                                    self.transformed_normals_, self.normals)

    @staticmethod
    def create(size: np.ndarray, density: float, is_static=False) -> "Cuboid":
        volume = float(np.prod(size))
        vertices = size * CUBE_VERTICES
        normals = CUBE_NORMALS

        if is_static:
            return Cuboid(size, vertices, normals)

        mass = density * volume
        inv_mass = 1 / mass
        # The moment of inertia of a cuboid a matrix with the diagonal
        # (mass / 12) * (sy^2 + sz^2), (mass / 12) * (sx^2 + sz^2),
        # (mass / 12) * (sx^2 + sy^2).
        # https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        sx, sy, sz = size
        iw = (mass / 12) * (sy ** 2 + sz ** 2)
        iy = (mass / 12) * (sx ** 2 + sz ** 2)
        iz = (mass / 12) * (sx ** 2 + sy ** 2)
        inertia = np.diag([iw, iy, iz]).astype(dtype=np.float64)
        inv_inertia = np.linalg.inv(inertia)
        return Cuboid(size, vertices, normals,
                      np.zeros(3, np.float64), np.zeros(3, np.float64),
                      mass, inv_mass,
                      inertia, inv_inertia)


RigidBody = Union[Sphere, Cuboid]


# the systems are defined by which components they operate over
Systems = {
    "physics": ["position", "rotation",
                "linear_velocity", "angular_velocity",
                "mass", "inertia_0"],
    "collision": ["aabb"]
}


@staticmethod
def create_body(config: RigidBodyConfig) -> RigidBody:
    # sample size, density, and kind
    size = np.array(config.size.sample_vec(), np.float64)
    density = config.density.sample()
    kind = choice(config.kinds)

    match kind:
        case PrimitiveKind.SPHERE:
            body = Sphere.create(size[0]/2, density)
        case PrimitiveKind.CUBOID:
            body = Cuboid.create(size, density, config.is_static)
        case _:
            raise ValueError(f"Unknown primitive kind: {kind}")

    # sample position, rotation
    body.move(np.array(config.position.sample_vec()))
    body.rotate(np.array(config.rotation.sample_vec()))
    if not config.is_static:
        # sample linear and angular velocities
        body.linear_velocity[:] = config.velocity.sample_vec()
        body.angular_velocity[:] = config.angular_velocity.sample_vec()

    # this name will be used as a key to the instance level
    body.name = f"{config.name}_{str(kind)}"

    for system, components in Systems.items():
        # check if the body has the system and store a flag
        has_system = all(hasattr(body, comp) for comp in components)
        setattr(body, system, has_system)

    return body
