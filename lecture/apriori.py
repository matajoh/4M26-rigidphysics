"""Demonstration of an a priori collision detection system."""

from colorsys import hls_to_rgb
import random
from typing import List, Mapping, NamedTuple, Tuple, Union

import decimal as d
import pygame

# The decimal type provides tuneable precision, and more accurate
# floating point arithmetic than the built-in float type.
dec = d.Decimal
decf = d.Decimal.from_float


class Vec2(NamedTuple("Vec2", [("x", d.Decimal), ("y", d.Decimal)])):
    """Standard 2D vector class."""
    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other: Union[int, d.Decimal]) -> "Vec2":
        return Vec2(self.x * other, self.y * other)

    def __rmul__(self, other: Union[int, d.Decimal]) -> "Vec2":
        return Vec2(self.x * other, self.y * other)

    def dot(self, other: "Vec2") -> Union[int, d.Decimal]:
        return self.x * other.x + self.y * other.y

    def project(self, other: "Vec2") -> "Vec2":
        return self * (self.dot(other) / self.dot(self))

    def normalize(self) -> "Vec2":
        return self * (1 / self.length())

    def is_zero(self) -> bool:
        return self.x == 0 and self.y == 0

    def length(self) -> Union[int, d.Decimal]:
        return self.norm(2)

    def norm(self, k=2) -> Union[int, d.Decimal]:
        if k == 1:
            return abs(self.x) + abs(self.y)
        elif k == 2:
            return (self.x**2 + self.y**2).sqrt()
        else:
            raise ValueError("k must be 1 or 2")

    @staticmethod
    def zero() -> "Vec2":
        return Vec2(0, 0)

    @staticmethod
    def random(x_min: int, x_max: int, y_min: int, y_max: int) -> "Vec2":
        return Vec2(random.randint(x_min, x_max),
                    random.randint(y_min, y_max))

    def dec(self) -> "Vec2":
        return Vec2(dec(self.x), dec(self.y))


INF = dec("Infinity")


class Circle(NamedTuple("Circle", [("n", int), ("pos", Vec2), ("vel", Vec2),
                                   ("radius", float), ("color", Tuple[int, int, int])])):
    """Circle class.
    
    Attributes:
        n: Circle number.
        pos: Position of the circle.
        vel: Velocity of the circle.
        radius: Radius of the circle.
        color: Color of the circle (RGB).
    """
    @staticmethod
    def random(n: int, h: float, width: int, height: int, rmin=5, rmax=30) -> "Circle":
        """Creates a random circle with a random color."""
        radius = random.randint(rmin, rmax)
        r, g, b = hls_to_rgb(h, 0.5, 1.0)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return Circle(n,
                      Vec2.random(radius, width - radius,
                                  radius, height - radius).dec(),
                      Vec2.random(-100, 100, -100, 100).dec(),
                      dec(radius),
                      color=(r, g, b))

    def draw(self, screen, border: int):
        pygame.draw.circle(screen, self.color,
                           (border + int(self.pos.x), border + int(self.pos.y)),
                           int(self.radius), 24)
        pygame.draw.circle(screen, "black",
                           (border + int(self.pos.x), border + int(self.pos.y)),
                           int(self.radius) - 22, 4)
        pygame.draw.circle(screen, "black",
                           (border + int(self.pos.x), border + int(self.pos.y)),
                           int(self.radius), 4)

    def update(self, dt: float) -> "Circle":
        """Move the state of the circle by dt seconds."""
        return self._replace(pos=self.pos + self.vel * dt)


class Wall(NamedTuple("Wall", [("horizontal", bool), ("value", d.Decimal)])):
    """Wall class.
    
    Attributes:
        horizontal: True if the wall is horizontal, False if vertical.
        value: The position of the wall.
    """
    def draw(self, screen: pygame.Surface, border: int):
        value = border + int(self.value)
        if self.horizontal:
            pygame.draw.line(screen, "black", (border, value),
                             (screen.get_width() - border, value),
                             4)
        else:
            pygame.draw.line(screen, "black", (value, border),
                             (value, screen.get_height() - border),
                             4)

    def next_collision(self, circle: Circle, eps=decf(-1e-10)) -> d.Decimal:
        """Calculates the time of the next collision with the wall."""
        if self.horizontal:
            pos = circle.pos.y
            vel = circle.vel.y
        else:
            pos = circle.pos.x
            vel = circle.vel.x

        if self.value < pos:
            gap = pos - circle.radius - self.value
            vel = -vel
        else:
            gap = self.value - pos - circle.radius

        if vel <= 0:
            return INF

        t = gap / vel
        if t < eps:
            return INF

        return t

    def collide_with(self, circle: Circle) -> Circle:
        """Calculates the new velocity of the circle after collision with the wall."""
        if self.horizontal:
            v0 = circle.vel._replace(y=-circle.vel.y)
        else:
            v0 = circle.vel._replace(x=-circle.vel.x)

        return circle._replace(vel=v0)


# NamedTuple for collision information
# time: Time of the collision
# i: Index of the first circle
# j: Index of the wall
Collision = NamedTuple("Collision", [("time", d.Decimal), ("i", int), ("j", int)])


def min_toi(tois: Mapping[Tuple[int, int], d.Decimal]) -> Collision:
    """Returns the collision with the smallest time of impact."""
    def toi(x):
        return x[1]

    item = min(tois.items(), key=toi)
    return Collision(toi(item), *item[0])


class PhysicsEngine:
    """Physics engine for the simulation."""
    def __init__(self, walls: List[Wall], circles: List[Circle]):
        self.walls = walls
        self.circles = circles
        self.time = dec(0)
        self.num_circles = len(circles)
        self.num_walls = len(walls)
        self.next_collision_: Collision = None
        self.initialise_tocs()

    def move(self, t: d.Decimal):
        """Move the simulation forward.
        
        Args:
            t: Time to move to.
        """
        assert t > self.time, "Time must be greater than current time"
        dt = t - self.time
        for i, ball in enumerate(self.circles):
            self.circles[i] = ball.update(dt)

        self.time = t

    def initialise_tocs(self):
        """Initialise the time of collision for all circles and walls."""
        self.tocs: Mapping[Tuple[int, int], d.Decimal] = {}

        for i in range(self.num_circles):
            b = self.circles[i]
            for j in range(self.num_walls):
                self.tocs[(i, j)] = self.walls[j].next_collision(b)

        self.next_collision_ = min_toi(self.tocs)

    def update_tocs(self, i: int):
        """Update the time of collision for a circle after a collision."""
        b = self.circles[i]
        for j in range(self.num_walls):
            self.tocs[(i, j)] = self.time + self.walls[j].next_collision(b)

        self.next_collision_ = min_toi(self.tocs)

    def update(self, dt: d.Decimal):
        """Move the simulation forward by dt seconds."""
        end_time = self.time + dt
        while True:
            coll = self.next_collision_
            if coll.time <= end_time:
                self.move(coll.time)
                bi = self.walls[coll.j].collide_with(self.circles[coll.i])
                self.circles[coll.i] = bi
                self.update_tocs(coll.i)
            else:
                break

        self.move(end_time)


class Simulation:
    """Out of scope."""
    def __init__(self, width=1020, height=740, num_circles=3,
                 rmin=80, rmax=100, border=50, debug=False):
        pygame.init()
        c = d.getcontext()
        c.prec = 60
        c.traps[d.FloatOperation] = True
        self.width = width
        self.height = height
        self.border = border
        self.debug = debug
        self.screen = pygame.display.set_mode((width + 2 * border, height + 2 * border))
        self.clock = pygame.time.Clock()
        self.running = False

        self.walls: List[Wall] = [
            Wall(True, dec(0)),
            Wall(True, dec(height)),
            Wall(False, dec(0)),
            Wall(False, dec(width))
        ]

        self.circles: List[Circle] = [Circle.random(i, i / num_circles, width, height, rmin, rmax)
                                      for i in range(num_circles)]

        self.engine = PhysicsEngine(self.walls, self.circles)
        self.font = pygame.font.SysFont(None, 24)

    def run(self):
        self.running = True
        dt = 0
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            self.screen.fill("white")

            self.engine.update(decf(dt))

            collision = self.engine.next_collision_
            circle = self.circles[collision.i].update(collision.time - self.engine.time)
            pygame.draw.circle(self.screen, circle.color,
                               (int(circle.pos.x + self.border), int(circle.pos.y + self.border)),
                               int(circle.radius), 8)

            for circle in self.circles:
                circle.draw(self.screen, self.border)

            for wall in self.walls:
                wall.draw(self.screen, self.border)

            if self.debug:
                fps = self.clock.get_fps()
                img = self.font.render(f"fps={fps:.2f}", True, "black")
                self.screen.blit(img, (0, 0))

            pygame.display.flip()
            dt = self.clock.tick(60) / 1000

        pygame.quit()


if __name__ == "__main__":
    display = Simulation()
    display.run()
