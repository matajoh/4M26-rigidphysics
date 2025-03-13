"""Module providing the simulation code.

NB This module is out of scope for the Tripos, but may be of interest
to students as a simpler interactive system (due to the 2D nature of
the graphics.)
"""

from colorsys import hls_to_rgb
import json
import math
import random
import time
import pygame
from pygame import Vector2

from .bodies import Circle, Polygon
from ..config import DetectionKind, Resolution
from .engine import PhysicsEngine, PhysicsMode


class Simulation:
    def __init__(self, resolution: Resolution,
                 physics_mode=PhysicsMode.FRICTION,
                 detection_kind=DetectionKind.QUADTREE,
                 debug=False, show_contacts=False, snapshot=False):
        pygame.init()
        self.width = resolution.width
        self.height = resolution.height
        self.debug = debug
        self.snapshot = snapshot
        self.screen = pygame.display.set_mode((resolution.width, resolution.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.running = False

        self.engine = PhysicsEngine(resolution.width, resolution.height, physics_mode,
                                    detection_kind, show_contacts)
        floor = Polygon.create_rectangle(30, 2, 2, "darkgreen", True).move_to(Vector2(0, 10))
        ledge0 = Polygon.create_rectangle(15, 1, 2, "gray", True).move_to(Vector2(-7.5, 0)).rotate_to(math.pi / 10)
        ledge1 = Polygon.create_rectangle(10, 1, 2, "darkred", True).move_to(Vector2(7, -5)).rotate_to(-math.pi / 10)
        self.engine.add_body(floor)
        self.engine.add_body(ledge0)
        self.engine.add_body(ledge1)

    def save_snapshot(self):
        path = f"snapshot_{time.strftime('%Y%m%d-%H%M%S')}"
        pygame.image.save(self.screen, f"{path}.png")
        bodies = [body.to_dict() for body in self.engine.bodies]
        with open(f"{path}.json", "w") as file:
            json.dump(bodies, file, indent=4)

    def run(self):
        self.running = True
        dt = 0
        physics_elapsed = 0
        samples = 0
        physics_stats = "Physics: ms"
        paused = False
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = self.engine.to_world(Vector2(event.pos))
                    if event.button == 1:
                        radius = random.uniform(1, 1.25)
                        h = random.uniform(0, 1)
                        r, g, b = hls_to_rgb(h, 0.5, 1.0)
                        color = int(r * 255), int(g * 255), int(b * 255)
                        self.engine.add_body(Circle.create(radius, 2, color).move_to(pos))
                    elif event.button == 3:
                        h = random.uniform(0, 1)
                        r, g, b = hls_to_rgb(h, 0.5, 1.0)
                        color = int(r * 255), int(g * 255), int(b * 255)
                        if random.random() < 0.25:
                            width = random.uniform(2, 3)
                            height = random.uniform(2, 3)
                            rectangle = Polygon.create_rectangle(width, height, 2, color)
                            self.engine.add_body(rectangle.move_to(pos))
                        else:
                            num_sides = random.randint(3, 8)
                            radius = random.uniform(1.25, 1.5)
                            polygon = Polygon.create_regular_polygon(num_sides, radius, 2, color)
                            self.engine.add_body(polygon.move_to(pos))

            self.screen.fill("white")

            start = time.perf_counter()
            if not paused:
                self.engine.step(dt)

            physics_elapsed += time.perf_counter() - start
            samples += 1
            if samples > 100:
                physics_stats = f"Physics: {physics_elapsed / samples * 1000:.2f} ms"
                physics_elapsed = 0
                samples = 0

            self.engine.draw(self.screen)

            if self.debug:
                text = self.font.render(f"FPS: {self.clock.get_fps():.2f}", True, "black")
                self.screen.blit(text, (10, 10))
                text = self.font.render(physics_stats, True, "black")
                self.screen.blit(text, (10, 30))
                text = self.font.render(f"Bodies: {len(self.engine.bodies)}", True, "black")
                self.screen.blit(text, (10, 50))

            pygame.display.flip()
            dt = self.clock.tick_busy_loop(60) / 1000

        if self.snapshot:
            self.save_snapshot()

        print("Shutting down...")
        start = time.perf_counter()
        pygame.quit()
        print(f"Shutdown took {time.perf_counter() - start:.2f} seconds")
