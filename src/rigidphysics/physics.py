from numba import jit
import numpy as np

from .bodies import RigidBody
from .config import PhysicsMode
from .collisions import Collision


@jit(cache=True)
def resolve_collision_basic(e: float,
                            a_lv: np.ndarray,
                            a_inv_mass: float,
                            b_lv: np.ndarray,
                            b_inv_mass: float,
                            n: np.ndarray):
    """Resolve a collision between two rigid bodies.
    
    Description:
        This basic version is purely focused on positional kinematics.
        It is useful for particle simulations or for debugging during
        early development.
    """
    relative_velocity = b_lv - a_lv

    if relative_velocity.dot(n) > 0:
        return

    j = -(1 + e) * relative_velocity.dot(n)
    j /= a_inv_mass + b_inv_mass
    impulse = j * n
    a_lv += -impulse * a_inv_mass
    b_lv += impulse * b_inv_mass


@jit(cache=True)
def resolve_collision_rotation(e: float,
                               a_p: np.ndarray,
                               a_lv: np.ndarray,
                               a_inv_mass: float,
                               a_av: np.ndarray,
                               a_inv_inertia: np.ndarray,
                               b_p: np.ndarray,
                               b_lv: np.ndarray,
                               b_inv_mass: float,
                               b_av: np.ndarray,
                               b_inv_inertia: np.ndarray,
                               n: np.ndarray,
                               impulse_list: np.ndarray,
                               ra_list: np.ndarray,
                               rb_list: np.ndarray,
                               contacts: np.ndarray):
    """Resolve a collision between two rigid bodies with rotation.
    
    Description:
        This is a halfway house between basic physics and full simulation.
        It is mostly useful for ablation testing the friction system, as
        rotation without friction results in...very weird physical
        possibilities.
    """
    contact_count = len(contacts)

    impulse_list[:] = 0
    ra_list[:] = 0
    rb_list[:] = 0

    for i in range(contact_count):
        ra = contacts[i] - a_p
        rb = contacts[i] - b_p

        ra_list[i] = ra
        rb_list[i] = rb

        angular_linear_velocity_a = np.cross(a_av, ra)
        angular_linear_velocity_b = np.cross(b_av, rb)
        relative_velocity = ((b_lv + angular_linear_velocity_b) -
                             (a_lv + angular_linear_velocity_a))
        contact_velocity_mag = np.dot(relative_velocity, n)

        ra_cross_n = np.cross(ra, n)
        rb_cross_n = np.cross(rb, n)
        denom = np.cross(a_inv_inertia @ ra_cross_n, ra)
        denom += np.cross(b_inv_inertia @ rb_cross_n, rb)
        denom = np.dot(denom, n) + a_inv_mass + b_inv_mass
        j = -(1 + e) * contact_velocity_mag
        j /= denom
        j /= contact_count

        impulse = j * n
        impulse_list[i] = impulse

    for i in range(contact_count):
        impulse = impulse_list[i]
        ra = ra_list[i]
        rb = rb_list[i]

        a_lv += -impulse * a_inv_mass
        a_av += -np.cross(ra, impulse) @ a_inv_inertia
        b_lv += impulse * b_inv_mass
        b_av += np.cross(rb, impulse) @ b_inv_inertia


@jit(cache=True)
def resolve_collision_rotation_friction(e: float,
                                        sf: float,
                                        df: float,
                                        a_p: np.ndarray,
                                        a_lv: np.ndarray,
                                        a_inv_mass: float,
                                        a_av: np.ndarray,
                                        a_inv_inertia: np.ndarray,
                                        b_p: np.ndarray,
                                        b_lv: np.ndarray,
                                        b_inv_mass: float,
                                        b_av: np.ndarray,
                                        b_inv_inertia: np.ndarray,
                                        n: np.ndarray,
                                        impulse_list: np.ndarray,
                                        ra_list: np.ndarray,
                                        rb_list: np.ndarray,
                                        j_list: np.ndarray,
                                        friction_impulse_list: np.ndarray,
                                        contacts: np.ndarray):
    """Resolve a collision between two rigid bodies with rotation and friction."""
    contact_count = len(contacts)

    impulse_list[:] = 0
    ra_list[:] = 0
    rb_list[:] = 0
    friction_impulse_list[:] = 0
    j_list[:] = 0

    for i in range(contact_count):
        ra = contacts[i] - a_p
        rb = contacts[i] - b_p

        ra_list[i] = ra
        rb_list[i] = rb

        angular_linear_velocity_a = np.cross(a_av, ra)
        angular_linear_velocity_b = np.cross(b_av, rb)
        rv = ((b_lv + angular_linear_velocity_b) -
              (a_lv + angular_linear_velocity_a))
        contact_velocity_mag = np.dot(rv, n)

        ra_cross_n = np.cross(ra, n)
        rb_cross_n = np.cross(rb, n)
        denom = np.cross(a_inv_inertia @ ra_cross_n, ra)
        denom += np.cross(b_inv_inertia @ rb_cross_n, rb)
        denom = np.dot(denom, n) + a_inv_mass + b_inv_mass
        j = -(1 + e) * contact_velocity_mag
        j /= denom
        j /= contact_count

        j_list[i] = j

        impulse = j * n
        impulse_list[i] = impulse

    for i in range(contact_count):
        impulse = impulse_list[i]
        ra = ra_list[i]
        rb = rb_list[i]

        a_lv += -impulse * a_inv_mass
        a_av += -np.cross(ra, impulse) @ a_inv_inertia
        b_lv += impulse * b_inv_mass
        b_av += np.cross(rb, impulse) @ b_inv_inertia

    for i in range(contact_count):
        ra = ra_list[i]
        rb = rb_list[i]
        a_angular_linear_velocity = np.cross(a_av, ra)
        b_angular_linear_velocity = np.cross(b_av, rb)
        rv = ((b_lv + b_angular_linear_velocity) -
              (a_lv + a_angular_linear_velocity))

        tangent = rv - np.dot(rv, n) * n
        tangent_length = np.linalg.norm(tangent)
        if tangent_length < 1e-3:
            continue

        tangent /= tangent_length
        ra_cross_t = np.cross(ra, tangent)
        rb_cross_t = np.cross(rb, tangent)
        jt = -np.dot(rv, tangent)
        denom = np.cross(a_inv_inertia @ ra_cross_t, ra)
        denom += np.cross(b_inv_inertia @ rb_cross_t, rb)
        denom = np.dot(denom, tangent) + a_inv_mass + b_inv_mass
        jt /= denom
        jt /= contact_count

        j = j_list[i]

        # Coulomb's law
        if abs(jt) <= j * sf:
            friction_impulse = jt * tangent
        else:
            friction_impulse = -j * tangent * df

        friction_impulse_list[i] = friction_impulse

    for i in range(contact_count):
        friction_impulse = friction_impulse_list[i]
        ra = ra_list[i]
        rb = rb_list[i]

        a_lv += -friction_impulse * a_inv_mass
        a_av += -np.cross(ra, friction_impulse) @ a_inv_inertia
        b_lv += friction_impulse * b_inv_mass
        b_av += np.cross(rb, friction_impulse) @ b_inv_inertia


class Physics:
    """Class providing the physics system.
    
    Description:
        This class primarily acts as a wrapper for the routines above,
        in particular by holding the pre-allocated buffers.
    """
    def __init__(self, mode: PhysicsMode, restitution: float,
                 static_friction: float, dynamic_friction: float):
        self.mode = mode
        self.restitution = restitution
        self.static_friction = static_friction
        self.dynamic_friction = dynamic_friction
        self.impulse_list = np.empty((25, 3), np.float64)
        self.ra_list = np.empty((25, 3), np.float64)
        self.rb_list = np.empty((25, 3), np.float64)
        self.friction_impulse_list = np.empty((25, 3), np.float64)
        self.j_list = np.empty(25, np.float64)
        self.lv_zero = np.zeros(3, np.float64)
        self.av_zero = np.zeros(3, np.float64)

    @property
    def needs_contacts(self) -> bool:
        return self.mode in (PhysicsMode.ROTATION, PhysicsMode.FRICTION)

    def resolve_collision(self, a: RigidBody, b: RigidBody,
                          collision: Collision, contacts: np.ndarray):
        normal = collision.normal
        a_linear_velocity = a.linear_velocity if a.physics else self.lv_zero
        a_angular_velocity = a.angular_velocity if a.physics else self.av_zero
        b_linear_velocity = b.linear_velocity if b.physics else self.lv_zero
        b_angular_velocity = b.angular_velocity if b.physics else self.av_zero
        match self.mode:
            case PhysicsMode.NONE:
                if a.physics:
                    a.linear_velocity -= normal * np.dot(a.linear_velocity, normal)
                if b.physics:
                    b.linear_velocity -= normal * np.dot(b.linear_velocity, normal)
            case PhysicsMode.BASIC:
                resolve_collision_basic(
                    self.restitution,
                    a_linear_velocity, a.inv_mass,
                    b_linear_velocity, b.inv_mass,
                    normal)
            case PhysicsMode.ROTATION:
                resolve_collision_rotation(
                    self.restitution,
                    a.position, a_linear_velocity, a.inv_mass,
                    a_angular_velocity, a.inv_inertia,
                    b.position, b_linear_velocity, b.inv_mass,
                    b_angular_velocity, b.inv_inertia,
                    normal, self.impulse_list, self.ra_list, self.rb_list,
                    contacts)
            case PhysicsMode.FRICTION:
                resolve_collision_rotation_friction(
                    self.restitution, self.static_friction, self.dynamic_friction,
                    a.position, a_linear_velocity, a.inv_mass,
                    a_angular_velocity, a.inv_inertia,
                    b.position, b_linear_velocity, b.inv_mass,
                    b_angular_velocity, b.inv_inertia,
                    normal, self.impulse_list, self.ra_list, self.rb_list,
                    self.j_list, self.friction_impulse_list, contacts)
            case _:
                raise ValueError("Invalid physics mode")
