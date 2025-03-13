"""Module providing the simulation window for the physics engine.

NB This module is extremely out of scope for the Tripos.
"""

from colorsys import hls_to_rgb
import ctypes
import math
import os
import random
import time

import numpy as np
import pyglet
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3, Vec4, Vec2
from pyglet.gl import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_COLOR_BUFFER_BIT,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_FLOAT,
    GL_MULTISAMPLE,
    GL_LINES,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_TRIANGLES,
    GL_STATIC_DRAW,
    GLfloat,
    GLuint,
    glBindBuffer,
    glBlendFunc,
    glBufferSubData,
    glClear,
    glClearColor,
    glEnable,
    glGenBuffers,
    glGenVertexArrays,
    glBindVertexArray,
    glBufferData,
    glEnableVertexAttribArray,
    glVertexAttribPointer,
    glDrawArrays,
    glDrawArraysInstanced,
    glVertexAttribDivisor,
)

from .collisions import collide
from .contacts import find_contact_points


from .config import PhysicsMode, RigidBodyConfig, SimulationConfig
from .physics import Physics
from .primitives import Lines, Primitive
from .engine import PhysicsEngine, PrimitiveSpan
from .bodies import create_body


def read_shader(name: str) -> str:
    shaders_path = os.path.join(os.path.dirname(__file__), "shaders")
    with open(os.path.join(shaders_path, name)) as f:
        return f.read()


def rot_from_euler_angles(angles: Vec3) -> Mat4:
    transform = Mat4.from_rotation(angles.x, Vec3(1, 0, 0))
    transform = transform.rotate(angles.y, Vec3(0, 1, 0))
    transform = transform.rotate(angles.z, Vec3(0, 0, 1))
    return transform


warmup_cube_config = RigidBodyConfig.from_dict({
    "name": "cuboid",
    "position": {
        "min": [-1, -1, -1],
        "max": [1, 1, 1]
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
    "rotation": 0,
})


warmup_sphere_config = RigidBodyConfig.from_dict({
    "name": "sphere",
    "position": {
        "min": [-1, -1, -1],
        "max": [1, 1, 1]
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
    "rotation": 0,
})


def precompile(mode: PhysicsMode, count=100, cube_prob=0.5):
    print("compiling...")
    contacts = np.empty((25, 3), np.float64)
    physics = Physics(mode, 0.5, 0.5, 0.5)
    for _ in range(count):
        a = create_body(warmup_cube_config if random.random() < cube_prob else warmup_sphere_config)
        b = create_body(warmup_cube_config if random.random() < cube_prob else warmup_sphere_config)
        collision = collide(a, b)
        if collision is None:
            continue

        num_contacts = find_contact_points(a, b, collision, contacts)
        if num_contacts == 0:
            continue

        physics.resolve_collision(a, b, collision, contacts[:num_contacts])

    print("done.")


class InstanceData:
    def __init__(self, engine: PhysicsEngine):
        self.engine = engine
        self.buffer = GLuint()
        self.num_instances = engine.num_instances
        self.size = engine.num_instances * PhysicsEngine.ELEMENTS_PER_INSTANCE * ctypes.sizeof(GLfloat)
        self.data = engine.instance_buffer.ctypes.data_as(ctypes.POINTER(GLfloat))

        glGenBuffers(1, self.buffer)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffer)
        glBufferData(GL_ARRAY_BUFFER, self.size, self.data, GL_DYNAMIC_DRAW)

    def update(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.buffer)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.size, self.data)


class MeshCloud:
    def __init__(self, instances: InstanceData, info: PrimitiveSpan, program: ShaderProgram):
        self.name = info.name
        self.primitive = info.primitive
        self.program = program
        self.num_verts = len(self.primitive.buffer) // Primitive.ELEMENTS_PER_VERTEX

        program.use()

        self.buffer = GLuint()
        glGenBuffers(1, self.buffer)

        self.gl_vao = GLuint()
        glGenVertexArrays(1, self.gl_vao)
        glBindVertexArray(self.gl_vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.buffer)
        size = self.num_verts * Primitive.ELEMENTS_PER_VERTEX
        data = self.primitive.buffer.ctypes.data_as(ctypes.POINTER(GLfloat))
        glBufferData(GL_ARRAY_BUFFER, size * ctypes.sizeof(GLfloat), data, GL_STATIC_DRAW)

        # set up vertex arrays
        glBindBuffer(GL_ARRAY_BUFFER, self.buffer)
        position = self.program.attributes["v_position"]["location"]
        glEnableVertexAttribArray(position)
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, 0)
        offset = self.num_verts * 3 * ctypes.sizeof(GLfloat)
        normal = self.program.attributes["v_normal"]["location"]
        glEnableVertexAttribArray(normal)
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 0, offset)
        offset += self.num_verts * 3 * ctypes.sizeof(GLfloat)
        uv = self.program.attributes["v_uv"]["location"]
        glEnableVertexAttribArray(uv)
        glVertexAttribPointer(uv, 2, GL_FLOAT, GL_FALSE, 0, offset)

        # set up instance arrays
        glBindBuffer(GL_ARRAY_BUFFER, instances.buffer)
        offset = info.start * 3 * ctypes.sizeof(GLfloat)
        instance_position = self.program.attributes["v_instance_position"]["location"]
        glEnableVertexAttribArray(instance_position)
        glVertexAttribPointer(instance_position, 3, GL_FLOAT, GL_FALSE, 0, offset)
        glVertexAttribDivisor(instance_position, 1)

        position_end = instances.num_instances * 3 * ctypes.sizeof(GLfloat)
        offset = position_end + info.start * 4 * ctypes.sizeof(GLfloat)
        instance_rotation = self.program.attributes["v_instance_rotation"]["location"]
        glEnableVertexAttribArray(instance_rotation)
        glVertexAttribPointer(instance_rotation, 4, GL_FLOAT, GL_FALSE, 0, offset)
        glVertexAttribDivisor(instance_rotation, 1)

        rotation_end = position_end + instances.num_instances * 4 * ctypes.sizeof(GLfloat)
        offset = rotation_end + info.start * 3 * ctypes.sizeof(GLfloat)
        instance_scale = self.program.attributes["v_instance_scale"]["location"]
        glEnableVertexAttribArray(instance_scale)
        glVertexAttribPointer(instance_scale, 3, GL_FLOAT, GL_FALSE, 0, offset)
        glVertexAttribDivisor(instance_scale, 1)

        scale_end = rotation_end + instances.num_instances * 3 * ctypes.sizeof(GLfloat)
        offset = scale_end + info.start * 3 * ctypes.sizeof(GLfloat)
        instance_color = self.program.attributes["v_instance_color"]["location"]
        glEnableVertexAttribArray(instance_color)
        glVertexAttribPointer(instance_color, 3, GL_FLOAT, GL_FALSE, 0, offset)
        glVertexAttribDivisor(instance_color, 1)

        color_end = scale_end + instances.num_instances * 3 * ctypes.sizeof(GLfloat)
        offset = color_end + info.start * ctypes.sizeof(GLfloat)
        instance_alpha = self.program.attributes["v_instance_alpha"]["location"]
        glEnableVertexAttribArray(instance_alpha)
        glVertexAttribPointer(instance_alpha, 1, GL_FLOAT, GL_FALSE, 0, offset)
        glVertexAttribDivisor(instance_alpha, 1)

    def draw(self, engine: PhysicsEngine):
        count = engine.primitive_queues[self.name].count
        self.program.use()
        self.program["u_checkerboard"] = self.primitive.checkerboard
        glBindVertexArray(self.gl_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffer)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self.num_verts, count)


class Wireframe:
    def __init__(self, lines: Lines, program: ShaderProgram, centroid=Vec3()):
        self.lines = lines
        self.color = Vec4(*lines.color, 1)
        self.program = program
        self.num_verts = lines.num_vertices
        self.centroid = centroid

        program.use()

        self.buffer = GLuint()
        glGenBuffers(1, self.buffer)

        self.gl_vao = GLuint()
        glGenVertexArrays(1, self.gl_vao)
        glBindVertexArray(self.gl_vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.buffer)
        size = self.num_verts * Primitive.ELEMENTS_PER_VERTEX
        data = lines.buffer.ctypes.data_as(ctypes.POINTER(GLfloat))
        glBufferData(GL_ARRAY_BUFFER, size * ctypes.sizeof(GLfloat), data, GL_STATIC_DRAW)

        # set up vertex arrays
        glBindBuffer(GL_ARRAY_BUFFER, self.buffer)
        position = self.program.attributes["v_position"]["location"]
        glEnableVertexAttribArray(position)
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, 0)

    def draw(self):
        self.program.use()
        self.program["u_color"] = self.color
        self.program["u_centroid"] = self.centroid
        glBindVertexArray(self.gl_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffer)
        glDrawArrays(GL_LINES, 0, self.num_verts)


def load_shader(name: str) -> ShaderProgram:
    vert_shader = Shader(read_shader(f"{name}.vert"), "vertex")
    frag_shader = Shader(read_shader(f"{name}.frag"), "fragment")
    return ShaderProgram(vert_shader, frag_shader)


ROTATE_SCALE = 0.2 * math.pi / 180
PAN_SCALE = 0.1
ZOOM_SCALE = 1.5


class Stopwatch:
    def __init__(self):
        self.start = time.perf_counter()
        self.frames_total = 0
        self.updates_total = 0
        self.physics_total = 0
        self.render_total = 0
        self.colliders_total = 0
        self.detections_total = 0

    def elapsed(self):
        return time.perf_counter() - self.start

    def reset(self):
        self.start = time.perf_counter()
        self.frames_total = 0
        self.updates_total = 0
        self.colliders_total = 0
        self.physics_total = 0
        self.render_total = 0
        self.detections_total = 0

    def add_update(self, collider_count: int, detection_count, physics_elapsed: float):
        self.updates_total += 1
        self.colliders_total += collider_count
        self.detections_total += detection_count
        self.physics_total += physics_elapsed

    def add_frame(self, render_elapsed):
        self.frames_total += 1
        self.render_total += render_elapsed

    def __repr__(self) -> str:
        colliders = self.colliders_total / self.updates_total
        detections = self.detections_total / self.updates_total
        physics_ms = 1000 * self.physics_total / self.updates_total
        render_ms = 1000 * self.render_total / self.frames_total
        fps = self.frames_total / self.elapsed()
        text = (f"DEBUG\nFPS: {fps:.2f}\n"
                f"physics: {physics_ms:.4f}\n"
                f"render: {render_ms:.4f}\n"
                f"#colliders: {colliders:.2f}\n"
                f"#detections: {detections:.2f}")
        return text


class Simulation(pyglet.window.Window):
    def __init__(self, sim_config: SimulationConfig, update_cb=None, click_cb=None):
        super().__init__(width=sim_config.resolution.width, height=sim_config.resolution.height,
                         resizable=False)
        precompile(sim_config.mode)
        self.sim_config = sim_config
        if self.sim_config.seed is not None:
            random.seed(self.sim_config.seed)

        glClearColor(1, 1, 1, 1)

        self.engine = PhysicsEngine(sim_config)
        self.update_cb = update_cb
        self.click_cb = click_cb
        self.default_camera_distance = sim_config.scale * 1.5
        self.setup_solids()
        self.setup_wireframes()
        self.update_projection()

        self.instances = InstanceData(self.engine)
        self.clouds = [MeshCloud(self.instances, span, self.solids_program)
                       for span in self.engine.primitives.values()
                       if span.render]

        self.cursor = sim_config.cursor
        if self.cursor.enabled:
            cursor = Lines.create_cursor(sim_config.cursor.color, sim_config.cursor.scale)
            self.cursor_wireframe = Wireframe(cursor, self.wireframe_program)
            self.cursor_position = Vec3()
        else:
            self.cursor_wireframe = None

        self.bounds = sim_config.bounds
        if self.bounds.enabled:
            bounds = Lines.create_cube(sim_config.bounds.color, sim_config.bounds.scale)
            self.bounds_wireframe = Wireframe(bounds, self.wireframe_program)
        else:
            self.bounds_wireframe = None

        self.camera_start = Vec3(0, 0, self.default_camera_distance)
        self.camera_angles = Vec3()
        self.cursor_delta = 0
        self.mouse_pos = Vec2()
        self.mv_inv = Mat4()
        self.transparent = False

        self.stats_display = pyglet.text.Label("", font_size=10, multiline=True,
                                               width=200, color=(0, 0, 0),
                                               x=5, y=90)
        self.stats = Stopwatch()

        self.update_mv()

        self.keyboard = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.keyboard)

        pyglet.clock.schedule_interval(lambda dt: self.update(dt), 1 / 120)
        self.set_mouse_visible(False)

    def setup_solids(self):
        self.solids_program = load_shader("solids")
        self.solids_program.use()
        self.solids_program["u_ambient_light_color"] = Vec3(0.5, 0.5, 0.5)
        self.solids_program["u_directional_light_color"] = Vec3(0.5, 0.5, 0.5)
        self.solids_program["u_directional_light_dir"] = Vec3(2.0, 1.0, 2.0)
        self.solids_program["u_transparent"] = 0
        self.solids_program["u_shading_type"] = 0

    def setup_wireframes(self):
        self.wireframe_program = load_shader("wireframe")

    def update_projection(self, fov_y_degrees=60):
        gl_prj_matrix = Mat4.perspective_projection(self.aspect_ratio, 0.1, 1000, fov_y_degrees)
        fov_y = fov_y_degrees * math.pi / 180
        focal_length_y = .5 * self.height / math.tan(.5 * fov_y)
        self.projection = Mat4(
            focal_length_y, 0, 0, 0,
            0, focal_length_y, 0, 0,
            self.width / 2, self.height / 2, 1, 0,
            0, 0, 0, 1,
        )
        self.unprojection = ~self.projection
        self.solids_program.use()
        self.solids_program["u_proj_matrix"] = gl_prj_matrix
        self.wireframe_program.use()
        self.wireframe_program["u_proj_matrix"] = gl_prj_matrix

    def update_cursor(self):
        if not self.cursor.enabled:
            return

        pos = Vec4(*self.mouse_pos, 1, 1)
        ray = self.unprojection @ pos
        ray = Vec3(ray.x, ray.y, -ray.z).normalize() * (self.camera_start.z + self.cursor_delta)
        cursor_position = self.mv_inv @ Vec4(*ray, 1)
        self.cursor_position = Vec3(*cursor_position[:3])
        self.cursor_wireframe.centroid = self.cursor_position

    def update_mv(self):
        rotate_mat = rot_from_euler_angles(self.camera_angles)
        translate_mat = Mat4.from_translation(-self.camera_start)
        mv = translate_mat @ rotate_mat
        self.mv_inv = ~mv
        n = self.mv_inv.transpose()
        self.solids_program.use()
        self.solids_program["u_mv_matrix"] = mv
        self.solids_program["u_norm_matrix"] = n

        self.wireframe_program.use()
        self.wireframe_program["u_mv_matrix"] = mv

        self.update_cursor()

    def move_cursor(self, delta: float):
        self.cursor_delta += delta
        self.update_cursor()

    def zoom(self, delta: float):
        scale = math.log(abs(self.camera_start.z) + 1.1) * ZOOM_SCALE
        self.camera_start += Vec3(0, 0, -delta) * scale
        self.update_mv()

    def rotate(self, delta_x: float, delta_y: float):
        self.camera_angles += Vec3(-delta_y, delta_x, 0) * ROTATE_SCALE
        self.update_mv()

    def spin(self, delta_x: float, delta_y: float):
        magnitude = (delta_x * delta_x + delta_y * delta_y) ** 0.5
        if delta_x > 0:
            magnitude = -magnitude

        self.camera_angles += Vec3(0, 0, magnitude) * ROTATE_SCALE
        self.update_mv()

    def pan(self, delta_x: float, delta_y: float):
        scale = -math.log(abs(self.camera_start.z) + 1.1) * PAN_SCALE
        self.camera_start += Vec3(delta_x, delta_y, 0) * scale
        self.update_mv()

    def on_resize(self, width, height):
        self.update_projection()

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_pos = Vec2(x, y)
        self.update_cursor()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.mouse_pos = Vec2(x, y)
        if modifiers & pyglet.window.key.MOD_CTRL:
            return

        if buttons & pyglet.window.mouse.LEFT:
            if modifiers & pyglet.window.key.MOD_SHIFT:
                self.pan(dx, dy)
            elif modifiers & pyglet.window.key.MOD_ALT:
                self.spin(dx, dy)
            else:
                self.rotate(dx, dy)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.zoom(scroll_y)

    def on_mouse_press(self, x, y, button, modifiers):
        if self.click_cb is not None:
            self.click_cb(x, y, button, modifiers)

        if modifiers & pyglet.window.key.MOD_CTRL:
            if self.cursor.enabled:
                sim_config = None
                if button == pyglet.window.mouse.LEFT:
                    sim_config = self.cursor.left_button
                elif button == pyglet.window.mouse.RIGHT:
                    sim_config = self.cursor.right_button

                if sim_config is not None:
                    body = create_body(sim_config)
                    body.move_to(self.cursor_position)
                    hsv = sim_config.color.sample_vec()
                    color = hls_to_rgb(*hsv)
                    self.engine.add_body(body, color)

    def on_key_press(self, symbol, modifiers):
        match symbol:
            case pyglet.window.key.ESCAPE:
                self.close()

            case pyglet.window.key.T:
                self.transparent = not self.transparent
                self.solids_program.use()
                self.solids_program["u_transparent"] = 1 if self.transparent else 0

            case pyglet.window.key.R:
                self.camera_start = Vec3(0, 0, self.default_camera_distance)
                self.camera_angles = Vec3()
                self.cursor_delta = 0
                self.update_mv()

    def on_draw(self):
        self.clear()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_MULTISAMPLE)
        if self.transparent:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        else:
            glEnable(GL_DEPTH_TEST)

        render_start = time.perf_counter()
        self.instances.update()
        render_elapsed = time.perf_counter() - render_start

        for cloud in self.clouds:
            cloud.draw(self.engine)

        if self.cursor.enabled:
            self.cursor_wireframe.draw()

        if self.bounds.enabled:
            self.bounds_wireframe.draw()

        if self.sim_config.debug:
            self.stats.add_frame(render_elapsed)
            self.stats_display.draw()

    def update(self, dt: float):
        if self.update_cb is not None:
            self.update_cb(dt, self.keyboard)

        if self.keyboard[pyglet.window.key.W]:
            self.move_cursor(1)
        elif self.keyboard[pyglet.window.key.S]:
            self.move_cursor(-1)

        physics_start = time.perf_counter()
        self.engine.step(dt)
        physics_elapsed = time.perf_counter() - physics_start

        if self.sim_config.debug:
            self.stats.add_update(len(self.engine.colliders),
                                  self.engine.num_detections,
                                  physics_elapsed)

            if self.stats.elapsed() > 1:
                self.stats_display.text = repr(self.stats)
                print(self.stats_display.text)
                self.stats.reset()

    def run(self):
        pyglet.app.run()
