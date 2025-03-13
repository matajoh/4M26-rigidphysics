"""Module providing configuration data structures for the physics engine.

There is a great advantage to driving simulation software via
configuration files. The first is reproducibility: you can
reproduce a simulation by running the same configuration file. It also
makes it easier to share and collaborate on simulations. Finally, it
makes it easier to experiment with different parameters. While it is
possible to have every possible parameter as a command line argument,
this can get unwieldy very quickly and it is very easy to make mistakes.

The pattern here, of having a structured class (like a NamedTuple) which
parses the JSON and provides default values, is a solid approach to
emulate.
"""

from enum import Enum, auto
import json
import math
from random import uniform
from typing import NamedTuple, Tuple, Union

from pyglet.math import Vec3


class PrimitiveKind(Enum):
    UNKNOWN = 0
    SPHERE = 1
    CUBOID = 2

    def __str__(self) -> str:
        match self:
            case PrimitiveKind.SPHERE:
                return "sphere"
            case PrimitiveKind.CUBOID:
                return "cuboid"
            case _:
                return "unknown"


class Uniform(NamedTuple("Uniform", [("min", float), ("max", float)])):
    """A uniform distribution between min and max."""
    def sample(self) -> float:
        """Sample a value from the distribution."""
        return uniform(self.min, self.max)

    def sample_vec(self) -> Vec3:
        """Sample a vector with the same value for each component."""
        v = self.sample()
        return Vec3(v, v, v)

    @staticmethod
    def create(data: Union[dict, float]) -> "Uniform":
        if isinstance(data, (float, int)):
            return Uniform(data, data)

        return Uniform(data["min"], data["max"])


class Uniform3(NamedTuple("Uniform3", [("min", Vec3), ("max", Vec3)])):
    """A uniform distribution for a 3D vector."""
    def sample_vec(self) -> Vec3:
        """Sample a vector from the distribution."""
        return Vec3(
            uniform(self.min.x, self.max.x),
            uniform(self.min.y, self.max.y),
            uniform(self.min.z, self.max.z)
        )

    @staticmethod
    def create(data: Union[dict, Vec3, float]) -> Union[Uniform, "Uniform3"]:
        if isinstance(data, (Vec3, list)):
            return Uniform3(Vec3(*data), Vec3(*data))

        if isinstance(data, (float, int)):
            return Uniform(data, data)

        vmin = data["min"]
        vmax = data["max"]

        if isinstance(vmin, (float, int)):
            if isinstance(vmax, (float, int)):
                return Uniform(vmin, vmax)

            vmin = (vmin, vmin, vmin)
        elif isinstance(vmax, float):
            vmax = (vmax, vmax, vmax)

        vmin = Vec3(*vmin)
        vmax = Vec3(*vmax)
        return Uniform3(vmin, vmax)


class RigidBodyConfig(NamedTuple("RigidBodyConfig", [("name", str),
                                                     ("kinds", Tuple[PrimitiveKind, ...]),
                                                     ("max_count", int),
                                                     ("size", Union[Uniform3, Uniform]),
                                                     ("position", Union[Uniform3, Uniform]),
                                                     ("velocity", Union[Uniform3, Uniform]),
                                                     ("rotation", Union[Uniform3, Uniform]),
                                                     ("angular_velocity", Union[Uniform3, Uniform]),
                                                     ("density", Union[Uniform]),
                                                     ("is_static", bool),
                                                     ("is_visible", bool),
                                                     ("color", Uniform3)])):
    """Configuration for a rigid body.
    
    Description:
        This configuration can be used to sample rigid bodies
        from a distribution of possible bodies defined by values ranges
        over the different parameters.
    """
    @staticmethod
    def from_dict(data: dict) -> "RigidBodyConfig":
        if data is None:
            return None

        name = data["name"]
        if "kinds" in data:
            kinds = tuple(PrimitiveKind[name.upper()] for name in data["kinds"])
        else:
            kinds = (PrimitiveKind[name.upper()],)

        max_count = data.get("max_count", 1)
        size = Uniform3.create(data.get("size", 1))
        position = Uniform3.create(data.get("position", 0))
        velocity = Uniform3.create(data.get("velocity", 0))
        rotation = Uniform3.create(data.get("rotation", 0))
        angular_velocity = Uniform3.create(data.get("angular_velocity", 0))
        density = Uniform.create(data.get("density", 1))
        is_static = data.get("is_static", False)
        is_visible = data.get("is_visible", True)
        if is_visible:
            color = Uniform3.create(data.get("color", {"min": [0, 0.5, 1], "max": [1, 0.5, 1]}))
        else:
            color = None

        return RigidBodyConfig(name, kinds, max_count, size, position,
                               velocity, rotation, angular_velocity,
                               density, is_static, is_visible, color)

    @staticmethod
    def defaults(scale: float) -> "RigidBodyConfig":
        size = Vec3(scale, scale / 10, scale)
        pos = Vec3(0, -scale / 2, 0)
        floor = RigidBodyConfig.from_dict({
            "name": "floor",
            "kinds": ["cuboid"],
            "size": {"min": size, "max": size},
            "position": {"min": pos, "max": pos},
            "color": [1/3, .2, 1],
            "is_static": True
        })

        size = Vec3(scale / 2, scale / 20, scale / 2)
        pos = Vec3(-scale/4, 0, 0)
        rotation = Vec3(0, 0, -math.pi / 10)
        ledge0 = RigidBodyConfig.from_dict({
            "name": "ledge0",
            "kinds": ["cuboid"],
            "size": {"min": size, "max": size},
            "position": {"min": pos, "max": pos},
            "rotation": rotation,
            "color": [0, 2/3, 0],
            "is_static": True
        })

        pos = Vec3(scale/4, scale / 4, 0)
        rotation = Vec3(0, 0, math.pi / 10)
        ledge1 = RigidBodyConfig.from_dict({
            "name": "ledge1",
            "kinds": ["cuboid"],
            "size": {"min": size, "max": size},
            "position": {"min": pos, "max": pos},
            "rotation": rotation,
            "color": [0, 1/3, 1],
            "is_static": True
        })

        return floor, ledge0, ledge1

    @staticmethod
    def contacts() -> "RigidBodyConfig":
        return RigidBodyConfig.from_dict({
            "name": "contacts",
            "kinds": ["cuboid"],
            "max_count": 400,
            "size": 0.3,
            "color": [0, 0.5, 1]
        })

    @staticmethod
    def bounds(size: float) -> "RigidBodyConfig":
        return RigidBodyConfig.from_dict({
            "name": "bounds",
            "kinds": ["cuboid"],
            "max_count": 6,
            "size": [size, 0.1, size],
            "is_static": True
        })

    @property
    def max_size(self) -> float:
        if isinstance(self.size, Uniform3):
            return max(self.size.max)

        if isinstance(self.size, Uniform):
            return self.size.max

        raise ValueError("Invalid size configuration")


class CursorConfig(NamedTuple("CursorConfig", [("enabled", bool),
                                               ("color", Vec3),
                                               ("scale", float),
                                               ("left_button", RigidBodyConfig),
                                               ("right_button", RigidBodyConfig)])):
    """Configuration for the cursor.

    Description:
        The cursor lives as a part of the simulation and can be used to
        add new bodies to the simulation. It is represented by two rigid
        body configurations, one for the left mouse button and one for the
        right mouse button which will be added on a mouse click.
    """
    @staticmethod
    def from_dict(data: dict) -> "CursorConfig":
        enabled = data.get("enabled", False)
        if not enabled:
            return CursorConfig(enabled, None, None, None, None)

        scale = data.get("scale", 1)
        color = Vec3(*data.get("color", (0, 0, 0)))
        left_button = RigidBodyConfig.from_dict(data["left_button"])
        right_button = RigidBodyConfig.from_dict(data["right_button"])
        return CursorConfig(enabled, color, scale, left_button, right_button)

    def default(max_instances: int) -> "CursorConfig":
        enabled = True
        color = Vec3(0, 0, 0)
        scale = 1
        left_button = RigidBodyConfig.from_dict({
            "name": "left_button",
            "kinds": ["sphere"],
            "max_count": max_instances,
            "size": {"min": 2, "max": 2.5},
            "density": 2
        })
        right_button = RigidBodyConfig.from_dict({
            "name": "right_button",
            "kinds": ["cuboid"],
            "max_count": max_instances,
            "size": {"min": [2, 2, 2], "max": [3, 3, 3]},
            "density": 2
        })
        return CursorConfig(enabled, color, scale, left_button, right_button)


class BoundsConfig(NamedTuple("BoundsConfig", [("enabled", bool),
                                               ("scale", float),
                                               ("color", Vec3)])):
    """Configuration for the bounds of the simulation."""
    @staticmethod
    def from_dict(data: dict) -> "BoundsConfig":
        enabled = data.get("enabled", False)
        if not enabled:
            return BoundsConfig(enabled, None, None)

        scale = data.get("scale", 4)
        color = Vec3(*data.get("color", (0, 0, 0)))
        return BoundsConfig(enabled, scale, color)

    def default() -> "BoundsConfig":
        return BoundsConfig(False, None, None)


class Resolution(NamedTuple("Resolution", [("width", int), ("height", int)])):
    """Resolution of the simulation window."""
    @staticmethod
    def from_string(name: str) -> "Resolution":
        parts = name.split("x")
        return Resolution(int(parts[0]), int(parts[1]))


class PhysicsMode(Enum):
    NONE = auto()
    BASIC = auto()
    ROTATION = auto()
    FRICTION = auto()


class DetectionKind(Enum):
    BASIC = 1
    SPATIAL_HASHING = 2
    QUADTREE = 3


class SimulationConfig(NamedTuple("SimulationConfig", [
    ("seed", int),
    ("resolution", Resolution),
    ("gravity", Vec3),
    ("show_contacts", bool),
    ("debug", bool),
    ("cursor", CursorConfig),
    ("bounds", BoundsConfig),
    ("bodies", Tuple[RigidBodyConfig, ...]),
    ("mode", PhysicsMode),
    ("detection", DetectionKind),
    ("scale", float),
    ("restitution", float),
    ("static_friction", float),
    ("dynamic_friction", float),
    ("spatial_cell_size", float)
])):
    """Configuration for the simulation.
    
    Args:
        seed (int): The seed for the random number generator
        resolution (Resolution): The resolution of the simulation window
        gravity (Vec3): The gravity vector
        show_contacts (bool): Whether to show contact points
        debug (bool): Whether to show debug information
        cursor (CursorConfig): The configuration for the cursor
        bounds (BoundsConfig): The configuration for the bounds of the simulation
        bodies (Tuple[RigidBodyConfig]): The configurations for the rigid bodies
        mode (PhysicsMode): The physics mode
        detection (DetectionKind): The collision detection algorithm
        scale (float): The scale of the simulation
        restitution (float): The coefficient of restitution (shared by all bodies)
        static_friction (float): The coefficient of static friction (shared by all bodies)
        dynamic_friction (float): The coefficient of dynamic friction (shared by all bodies)
    """
    @staticmethod
    def from_dict(data: dict) -> "SimulationConfig":
        seed = data.get("seed", None)
        resolution = Resolution(*data.get("resolution", (1440, 1080)))
        gravity = Vec3(*data.get("gravity", (0, -9.81, 0)))
        debug = data.get("debug", False)
        show_contacts = data.get("show_contacts", False)
        if "cursor" in data:
            cursor = CursorConfig.from_dict(data["cursor"])
        else:
            cursor = CursorConfig.default(data.get("max_instances", 100))

        scale = data.get("scale", 30)
        if "bodies" in data:
            bodies = tuple(RigidBodyConfig.from_dict(body) for body in data.get("bodies", []))
        else:
            bodies = RigidBodyConfig.defaults(scale)

        mode = PhysicsMode[data.get("mode", "friction").upper()]
        if "bounds" in data:
            bounds = BoundsConfig.from_dict(data["bounds"])
        else:
            bounds = BoundsConfig.default()

        detection = DetectionKind[data.get("detection", "spatial_hashing").upper()]

        restitution = data.get("restitution", 0.5)
        static_friction = data.get("static_friction", 0.5)
        dynamic_friction = data.get("dynamic_friction", 0.5)
        spatial_cell_size = data.get("spatial_cell_size", 4)

        if show_contacts:
            bodies = (RigidBodyConfig.contacts(),) + bodies

        if cursor.enabled:
            bodies = bodies + (cursor.left_button, cursor.right_button)

        return SimulationConfig(
            seed, resolution, gravity, show_contacts, debug, cursor,
            bounds, bodies, mode, detection, scale, restitution,
            static_friction, dynamic_friction, spatial_cell_size)

    @staticmethod
    def create(resolution=Resolution(1440, 1080),
               show_contacts=False,
               debug=False,
               max_instances=100,
               mode="friction") -> "SimulationConfig":
        return SimulationConfig.from_dict({
            "resolution": resolution,
            "show_contacts": show_contacts,
            "debug": debug,
            "max_instances": max_instances,
            "mode": mode
        })

    @staticmethod
    def load(path: str) -> "SimulationConfig":
        with open(path, "r") as file:
            data = json.load(file)

        return SimulationConfig.from_dict(data)
