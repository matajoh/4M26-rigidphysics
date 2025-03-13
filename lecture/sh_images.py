"""Script which produces vector graphics visualisations of spatial hashing.

Out of scope for the Tripos.
"""

from colorsys import hls_to_rgb
import random
from typing import Callable, List, NamedTuple, Tuple
import drawsvg as draw
from pygame import Vector2


def save_png(filename: str, d: draw.Drawing):
    d.save_png(filename + ".png")


def save_svg(filename: str, d: draw.Drawing):
    d.save_svg(filename + ".svg")


def hue_to_color(h: float) -> str:
    r, g, b = hls_to_rgb(h, 0.5, 1)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


class Circle(NamedTuple("Circle", [("x", float), ("y", float), ("r", float)])):
    def intersects(self, other: "Circle") -> bool:
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 < (self.r + other.r) ** 2

    def move(self, x: float, y: float):
        return Circle(self.x + x, self.y + y, self.r)

    @staticmethod
    def create(width: float, height: float, max_radius: float):
        x = random.uniform(max_radius-width/2, width/2 - max_radius)
        y = random.uniform(max_radius-height/2, height/2 - max_radius)
        r = random.uniform(max_radius / 2, max_radius)
        return Circle(x, y, r)


def circles(save_drawing: Callable[[str, draw.Drawing], None], width=800, height=600,
            max_radius=20, count=400, seed=20080524):
    random.seed(seed)
    circles = []
    while len(circles) < count:
        collision = False
        c0 = Circle.create(width, height, max_radius)
        for c1 in circles:
            if c0.intersects(c1):
                collision = True
                break

        if not collision:
            circles.append(c0)

    d = draw.Drawing(width, height, origin="center")
    for c in circles:
        color = hue_to_color(random.uniform(0, 1))
        d.append(draw.Circle(c.x, c.y, c.r, fill=color, stroke="black", stroke_width="2"))

    save_drawing("sh_circles", d)

    cell_size = max_radius * 2
    num_columns = int(width / cell_size) + 1
    num_rows = int(height / cell_size) + 1
    for r in range(num_rows):
        y = (r - num_rows // 2 + 0.5) * cell_size
        d.append(draw.Line(-width/2, y, width/2, y, stroke="gray", stroke_width="1"))

    for c in range(num_columns):
        x = (c - num_columns // 2) * cell_size
        d.append(draw.Line(x, -height/2, x, height/2, stroke="gray", stroke_width="1"))

    save_drawing("sh_grid", d)

    d.view_box = (-cell_size, -cell_size * 0.5, cell_size * 3, cell_size * 3)

    save_drawing("sh_grid_zoom", d)


Cell = NamedTuple("Cell", [("r", int), ("c", int), ("color", str),
                           ("points", List[Tuple[float, float]])])


def hash(save_drawing: Callable[[str, draw.Drawing], None], num_rows=4,
         num_columns=6, cell_size=100, num_occupied=4, seed=20080524):
    random.seed(seed)
    width = num_columns * cell_size
    height = num_rows * cell_size
    d = draw.Drawing(width, height)
    d.view_box = (-1, -1, width + 2, height + 2)
    for r in range(num_rows + 1):
        y = r * cell_size
        d.append(draw.Line(0, y, width, y, stroke="gray", stroke_width="3"))

    for c in range(num_columns + 1):
        x = c * cell_size
        d.append(draw.Line(x, 0, x, height, stroke="gray", stroke_width="3"))

    occupied_cells = set()
    while len(occupied_cells) < num_occupied:
        r = random.randint(0, num_rows - 1)
        c = random.randint(0, num_columns - 1)
        occupied_cells.add((r, c))

    cells = {}
    radius = cell_size / 4
    for i, (r, c) in enumerate(occupied_cells):
        x = c * cell_size
        y = r * cell_size
        num_points = random.randint(1, 2)
        points = []
        while len(points) < num_points:
            c0 = Circle.create(cell_size, cell_size, radius).move(x + cell_size / 2, y + cell_size / 2)
            collision = False
            for c1 in points:
                if c0.intersects(c1):
                    collision = True
                    break

            if not collision:
                points.append(c0)

        color = hue_to_color(i / num_occupied)
        cells[(c, r)] = Cell(r, c, color, points)
        d.append(draw.Rectangle(x, y, cell_size, cell_size, fill=color, fill_opacity=0.25))
        for p in points:
            d.append(draw.Circle(p.x, p.y, p.r, fill="rgb(80, 80, 80)"))

    save_drawing("sh_hash_grid", d)

    cell_size /= 2
    half_size = cell_size / 2

    num_array = num_rows * num_columns
    d = draw.Drawing(num_array * cell_size, cell_size * 4)
    d.view_box = (-1, -1, d.width + 2, d.height + 2)
    for i in range(num_array):
        r = i % num_rows
        c = i // num_rows
        x = i * cell_size
        d.append(draw.Rectangle(x, 0, cell_size, cell_size, fill="none", stroke="gray", stroke_width="1"))
        if c > 0 and r == 0:
            d.append(draw.Line(x, 0, x, cell_size, stroke="rgb(80, 80, 80)", stroke_width="3"))

    def arrow(start: Vector2, end: Vector2):
        n = end - start
        np = Vector2(-n.y, n.x)
        line_end = start + n * 0.6
        d.append(draw.Line(start.x, start.y, line_end.x, line_end.y, stroke="rgb(80, 80, 80)", stroke_width="4"))
        t = [end, line_end + 0.3 * np, line_end - 0.3 * np]
        d.append(draw.Lines(t[0].x, t[0].y, t[1].x, t[1].y, t[2].x, t[2].y, close=True, fill="rgb(80, 80, 80)"))

    bodies = []
    for cell in cells.values():
        r, c, color, points = cell
        x = (c * num_rows + r) * cell_size
        d.append(draw.Rectangle(x, 0, cell_size, cell_size, fill=color, fill_opacity=0.25))
        y = cell_size
        for point in points:
            arrow(Vector2(x + half_size, y), Vector2(x + half_size, y + half_size))
            y += half_size
            bodies.append((c, r, point))
            d.append(draw.Rectangle(x, y, cell_size, cell_size, fill="none", stroke="gray", stroke_width="1"))
            d.append(draw.Text(str(len(bodies)), half_size * 1.25, x + half_size - 0.2, y + half_size - 0.2,
                               fill="black", text_anchor="middle", alignment_baseline="middle",
                               font_family="courier", font_weight="bold"))
            y += cell_size

    save_drawing("sh_hash_array", d)

    d.clear()
    cumulative_sum = []
    num_bodies = 0
    cell = None
    for i in range(num_array):
        r = i % num_rows
        c = i // num_rows
        x = i * cell_size
        d.append(draw.Rectangle(x, 0, cell_size, cell_size, fill="none", stroke="gray", stroke_width="1"))
        cumulative_sum.append(num_bodies)
        if (c, r) in cells:
            cell = cells[(c, r)]
            d.append(draw.Rectangle(x, 0, cell_size, cell_size, fill=cell.color, fill_opacity=0.25))

        d.append(draw.Text(str(num_bodies), half_size * 1.25, x + half_size - 0.2, half_size - 0.2,
                           fill="black", text_anchor="middle", alignment_baseline="middle",
                           font_family="courier", font_weight="bold"))

        if cell:
            num_bodies += len(cell.points)
            cell = None

        if c > 0 and r == 0:
            d.append(draw.Line(x, 0, x, cell_size, stroke="rgb(80, 80, 80)", stroke_width="3"))

    y = cell_size + half_size
    num_bodies = 0
    packed_colors = []
    for cell in cells.values():
        i = cell.c * num_rows + cell.r
        x = cumulative_sum[i] * cell_size
        for point in cell.points:
            num_bodies += 1
            d.append(draw.Rectangle(x, y, cell_size, cell_size, fill=cell.color, fill_opacity=0.25,
                                    stroke="gray", stroke_width="1"))
            d.append(draw.Text(str(num_bodies), half_size * 1.25, x + half_size - 0.2, y + half_size - 0.2,
                               fill="black", text_anchor="middle", alignment_baseline="middle",
                               font_family="courier", font_weight="bold"))
            x += cell_size

    for cell in sorted(cells.values(), key=lambda c: c.c * num_rows + c.r):
        for _ in cell.points:
            packed_colors.append(cell.color)

    save_drawing("sh_hash_final", d)

    d = draw.Drawing(num_array * cell_size, cell_size)
    cumulative_sum = []
    num_bodies = 0
    cell = None
    for i in range(num_array):
        r = i % num_rows
        c = i // num_rows
        x = i * cell_size
        d.append(draw.Rectangle(x, 0, cell_size, cell_size, fill="none", stroke="gray", stroke_width="1"))
        if (c, r) in cells:
            cell = cells[(c, r)]
            v = len(cell.points)
            num_bodies += v
            d.append(draw.Rectangle(x, 0, cell_size, cell_size, fill=cell.color, fill_opacity=0.25))
        else:
            v = 0

        d.append(draw.Text(str(v), half_size * 1.25, x + half_size - 0.2, half_size - 0.2,
                           fill="black", text_anchor="middle", alignment_baseline="middle",
                           font_family="courier", font_weight="bold"))

        cumulative_sum.append(num_bodies)
        if c > 0 and r == 0:
            d.append(draw.Line(x, 0, x, cell_size, stroke="rgb(80, 80, 80)", stroke_width="3"))

    save_drawing("sh_hash_counts", d)

    d.clear()
    for i, s in enumerate(cumulative_sum):
        r = i % num_rows
        c = i // num_rows
        x = i * cell_size
        d.append(draw.Rectangle(x, 0, cell_size, cell_size, fill="none", stroke="gray", stroke_width="1"))
        if (c, r) in cells:
            cell = cells[(c, r)]
            d.append(draw.Rectangle(x, 0, cell_size, cell_size, fill=cell.color, fill_opacity=0.25))

        d.append(draw.Text(str(s), half_size * 1.25, x + half_size - 0.2, half_size - 0.2,
                           fill="black", text_anchor="middle", alignment_baseline="middle",
                           font_family="courier", font_weight="bold"))

        if c > 0 and r == 0:
            d.append(draw.Line(x, 0, x, cell_size, stroke="rgb(80, 80, 80)", stroke_width="3"))

    save_drawing("sh_hash_cumulative", d)

    d = draw.Drawing(cell_size * (num_array + len(bodies) + 1), cell_size)
    d.view_box = (-1, -1, d.width + 2, d.height + 2)
    packed = [None] * len(bodies)

    for b, (bc, br, _) in enumerate(bodies):
        d.clear()
        # draw all the cells
        # decrement
        bi = bc * num_rows + br
        cumulative_sum[bi] -= 1
        packed[cumulative_sum[bi]] = b + 1

        for i, s in enumerate(cumulative_sum):
            r = i % num_rows
            c = i // num_rows
            x = i * cell_size
            if c > 0 and r == 0:
                d.append(draw.Line(x, 0, x, cell_size, stroke="rgb(80, 80, 80)", stroke_width="3"))

            d.append(draw.Rectangle(x, 0, cell_size, cell_size, fill="none", stroke="gray", stroke_width=1))
            if (c, r) in cells:
                cell = cells[(c, r)]
                d.append(draw.Rectangle(x, 0, cell_size, cell_size, fill=cell.color, fill_opacity=0.25))

            d.append(draw.Text(str(s), half_size * 1.25, x + half_size - 0.2, half_size - 0.2,
                               fill="black", text_anchor="middle", alignment_baseline="middle",
                               font_family="courier", font_weight="bold"))

        d.append(draw.Rectangle(bi * cell_size, 0, cell_size, cell_size, fill="none", stroke="red", stroke_width=3))

        x = num_array * cell_size + cell_size
        for i, color in enumerate(packed_colors):
            d.append(draw.Rectangle(x, 0, cell_size, cell_size, fill=color, fill_opacity=0.25,
                                    stroke="gray", stroke_width="1"))
            if packed[i] is not None:
                d.append(draw.Text(str(packed[i]), half_size * 1.25, x + half_size - 0.2, half_size - 0.2,
                                   fill="black", text_anchor="middle", alignment_baseline="middle",
                                   font_family="courier", font_weight="bold"))

            x += cell_size

        save_drawing(f"sh_hash_build_{b}", d)


if __name__ == "__main__":
    circles(save_svg)
    hash(save_svg)
