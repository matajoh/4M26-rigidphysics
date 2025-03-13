"""Script producing an animation of the GJK algorithm.

Description:
    This script contains an implementation of the Gilbert-Johnson-Keerthi (GJK) algorithm for
    two dimensions. All of the vector graphics code is out of scope for the Tripos.
"""

import random
from typing import Callable, List, Tuple

from convex_hull import compute_convex_hull
import drawsvg as draw
from pygame import Vector2


from rigidphysics.flat.bodies import Polygon
from rigidphysics.flat.contacts import point_segment_distance


def save_png(filename: str, d: draw.Drawing):
    d.save_png(filename + ".png")


def save_svg(filename: str, d: draw.Drawing):
    d.save_svg(filename + ".svg")


def minkowski_difference(a: List[Vector2], b: List[Vector2]) -> List[Vector2]:
    """Compute the minkowski diffrence between two convex polygons."""
    return [ai - bi for ai in a for bi in b]


def coords(vertices: List[Vector2], center: Vector2) -> List[float]:
    """Convert a list of vertices and a center to a list of coordinates."""
    return [v for vertex in vertices for v in vertex + center]


def draw_axis(d: draw.Drawing):
    """Out of scope."""
    half_width = d.width / 4
    border = 0.1
    d.append(draw.Line(border - half_width * 2, 0, -border, 0, stroke="gray", stroke_width="0.1"))
    d.append(draw.Line(-half_width, border - half_width, -half_width,
             half_width-border, stroke="gray", stroke_width="0.1"))
    d.append(draw.Circle(-half_width, 0, 0.2, fill="gray"))


def draw_polygons(d: draw.Drawing, a: Polygon, b: Polygon, md: List[Vector2]):
    """Out of scope."""
    half_width = d.width / 4
    left = Vector2(-half_width, 0)
    right = Vector2(half_width, 0)

    d.append(draw.Lines(*coords(a.transformed_vertices, right), close=True, fill="none",
                        stroke=f"rgb{a.color}", stroke_width="0.1"))
    d.append(draw.Lines(*coords(b.transformed_vertices, right), close=True, fill="none",
                        stroke=f"rgb{b.color}", stroke_width="0.1"))
    if md:
        d.append(draw.Lines(*coords(md, left), close=True, fill="none", stroke="black", stroke_width="0.1"))


def draw_simplex(d: draw.Drawing, simplex: List[Vector2]):
    """Out of scope."""
    half_width = d.width / 4
    left = Vector2(-half_width, 0)
    d.append(draw.Lines(*coords(simplex, left), close=True, fill="none", stroke="blue", stroke_width="0.1"))


def draw_voronoi(d: draw.Drawing, simplex: List[Vector2]):
    """Out of scope."""
    half_width = d.width / 4
    left = Vector2(-half_width, 0)

    a, b, c, = simplex
    ab = b - a

    clip_path = draw.ClipPath()
    clip_path.append(draw.Rectangle(-d.width / 2 + 0.1, -d.height / 2 + 0.1, d.width / 2 - .2, d.height - .2))

    ab_n = Vector2(-ab.y, ab.x) * 100
    square = [a, b, b + ab_n, a + ab_n]
    d.append(draw.Lines(*coords(square, left), close=True, fill="blue", fill_opacity="0.2",
                        stroke="gray", stroke_width="0.01", clip_path=clip_path))

    bc = c - b
    bc_n = Vector2(-bc.y, bc.x) * 100
    square = [b, c, c + bc_n, b + bc_n]
    d.append(draw.Lines(*coords(square, left), close=True, fill="blue", fill_opacity="0.2",
                        stroke="gray", stroke_width="0.01", clip_path=clip_path))

    ca = a - c
    ca_n = Vector2(-ca.y, ca.x) * 100
    square = [c, a, a + ca_n, c + ca_n]
    d.append(draw.Lines(*coords(square, left), close=True, fill="blue", fill_opacity="0.2",
                        stroke="gray", stroke_width="0.01", clip_path=clip_path))

    triangle = [a, a + ab_n, a + ca_n]
    d.append(draw.Lines(*coords(triangle, left), close=True, fill="red", fill_opacity="0.2",
                        stroke="gray", stroke_width="0.01", clip_path=clip_path))
    triangle = [b, b + bc_n, b + ab_n]
    d.append(draw.Lines(*coords(triangle, left), close=True, fill="red", fill_opacity="0.2",
                        stroke="gray", stroke_width="0.01", clip_path=clip_path))
    triangle = [c, c + ca_n, c + bc_n]
    d.append(draw.Lines(*coords(triangle, left), close=True, fill="red", fill_opacity="0.2",
                        stroke="gray", stroke_width="0.01", clip_path=clip_path))

    triangle = [a, b, c]
    d.append(draw.Lines(*coords(triangle, left), close=True, fill="green", fill_opacity="0.2",
                        stroke="black", stroke_width="0.1"))

    d.append(draw.Circle(a.x + left.x, a.y + left.y, 0.1, fill="black"))
    d.append(draw.Text("A", .5, a.x + left.x - 0.4, a.y + left.y + 0.4, fill="black"))
    d.append(draw.Circle(b.x + left.x, b.y + left.y, 0.1, fill="black"))
    d.append(draw.Text("B", .5, b.x + left.x + 0.2, b.y + left.y + 0.1, fill="black"))
    d.append(draw.Circle(c.x + left.x, c.y + left.y, 0.1, fill="black"))
    d.append(draw.Text("C", .5, c.x + left.x - 0.2, c.y + left.y - 0.2, fill="black"))

    rac = (a + c) * 0.5 + ca_n / 150
    d.append(draw.Text("RAC", .5, rac.x + left.x, rac.y + left.y, fill="black"))
    rbc = (b + c) * 0.5 + bc_n / 120
    d.append(draw.Text("RBC", .5, rbc.x + left.x - 0.3, rbc.y + left.y, fill="black"))


def draw_normal(d: draw.Drawing, index: int, n: Vector2, ns: Vector2, color: str,
                a_verts: List[Vector2], b_verts: List[Vector2], hide_arrow=False):
    """Out of scope."""
    half_width = d.width / 4
    left = Vector2(-half_width, 0)
    right = Vector2(half_width, 0)
    a = farthest_vertex(a_verts, n)
    b = farthest_vertex(b_verts, -n)
    sp = a - b
    ns = ns + left
    if not hide_arrow:
        ne = ns + 1.5 * n
        d.append(draw.Text(f"d{index}", x=ne.x, y=ne.y, font_size=0.4, font_weight="bold", fill=color,
                           text_anchor="middle", dominant_baseline="middle"))
        d.append(draw.Line(ns.x, ns.y, ns.x + 0.9 * n.x, ns.y + 0.9 * n.y, stroke=color, stroke_width="0.1"))
        np = Vector2(-n.y, n.x)
        triangle = [ns + 1.2 * n, ns + 0.9 * n + 0.2 * np, ns + 0.9 * n - 0.2 * np]
        d.append(draw.Lines(*coords(triangle, Vector2()), close=True, fill=color))
    d.append(draw.Circle(a.x + right.x, a.y + right.y, 0.15, fill=color, stroke="gray", stroke_width="0.05"))
    d.append(draw.Circle(b.x + right.x, b.y + right.y, 0.15, fill=color, stroke="gray", stroke_width="0.05"))
    d.append(draw.Circle(sp.x + left.x, sp.y + left.y, 0.15, fill=color, stroke="gray", stroke_width="0.05"))
    sp_text = sp + left + n * 0.5
    d.append(draw.Text(f"v{index}", x=sp_text.x, y=sp_text.y, font_size=0.4, font_weight="bold", fill=color,
                       text_anchor="middle", dominant_baseline="middle"))


def farthest_vertex(verts: List[Vector2], direction: Vector2) -> Vector2:
    """Return the farthest vertex in the provided direction."""
    max_dot = verts[0].dot(direction)
    max_vertex = verts[0]
    for v in verts[1:]:
        dot = v.dot(direction)
        if dot > max_dot:
            max_dot = dot
            max_vertex = v

    return max_vertex


def support_function(a_verts: List[Vector2], b_verts: List[Vector2], direction: Vector2) -> Vector2:
    """Produce a support point on the Minkowski difference of two convex polygons.

    Args:
        a_verts: Vertices of the first polygon.
        b_verts: Vertices of the second polygon.
        direction: The direction in which to compute the support point.
    Returns:
        The support point in the Minkowski difference.
    """
    a = farthest_vertex(a_verts, direction)
    b = farthest_vertex(b_verts, -direction)
    return a - b


def get_closest_edge(a: Vector2, b: Vector2, c: Vector2) -> Tuple[Vector2, Vector2]:
    """Returns the closest edge to the origin from the triangle.

    Args:
        a: First vertex of the triangle.
        b: Second vertex of the triangle.
        c: Third vertex of the triangle.
    Returns:
        The closest edge to the origin from the triangle as a tuple of two vertices.
    """
    closest = a, b
    origin = Vector2()
    min_dist = point_segment_distance(origin, a, b)
    d = point_segment_distance(origin, b, c)
    if d < min_dist:
        min_dist = d
        closest = b, c
    d = point_segment_distance(origin, c, a)
    if d < min_dist:
        closest = c, a

    return closest


def normal_direction_vector(a: Vector2, b: Vector2) -> Vector2:
    """Compute the normal direction vector from two points.

    Args:
         a: First point.
         b: Second point.
    Returns:
        A normalized vector perpendicular to the line segment from a to b.
    """
    return Vector2(-b.y + a.y, b.x - a.x).normalize()


def triple_product(a: Vector2, b: Vector2, c: Vector2) -> float:
    """Compute the triple product of three vectors.

    Args:
        a: First vector.
        b: Second vector.
        c: Third vector.
    Returns:
        The result of the triple product a x (b x c)
    """
    return a.dot(c) * b - a.dot(b) * c


def main(save_drawing: Callable[[str, draw.Drawing], None], width=16, height=9, seed=20080524):
    random.seed(seed)
    a = Polygon.create_regular_polygon(3, 1.3, 2, (255, 0, 0))
    b = Polygon.create_regular_polygon(5, 1.5, 2, (0, 255, 0))
    a.rotate_to(0.5)
    b.rotate_to(0.2)
    a.move_to(Vector2(0.35, 0.5))
    b.move_to(Vector2(-0.5, -0.5))
    md = minkowski_difference(a.transformed_vertices, b.transformed_vertices)
    md = compute_convex_hull(md)
    md_center = Vector2(sum(v.x for v in md) / len(md), sum(v.y for v in md) / len(md))
    d = draw.Drawing(width, height, origin="center")
    d.set_render_size(w=width*100)

    draw_axis(d)
    draw_polygons(d, a, b, md)
    save_drawing("gjk0", d)

    a_verts = a.transformed_vertices
    b_verts = b.transformed_vertices

    # First Iteration of GJK
    # Find the support point in the direction of the negative x-axis
    d1 = Vector2(-1, 0)
    sp1 = support_function(a_verts, b_verts, d1)

    # Find the normal direction vector from the support point to the origin
    # and compute the support point in that direction
    d2 = -sp1.normalize()
    sp2 = support_function(a_verts, b_verts, d2)

    # Compute the normal direction vector from sp1 to sp2
    # and compute the support point in that direction
    d3 = normal_direction_vector(sp1, sp2)
    sp3 = support_function(a_verts, b_verts, d3)

    simplex = (sp1, sp2, sp3)

    d.clear()
    draw_axis(d)
    draw_polygons(d, a, b, md)
    draw_simplex(d, simplex)
    draw_normal(d, 1, d1, md_center + Vector2(2, 0), "cyan", a_verts, b_verts)
    draw_normal(d, 2, d2, sp1, "magenta", a_verts, b_verts)
    draw_normal(d, 3, d3, 0.5 * (sp1 + sp2), "orange", a_verts, b_verts)
    save_drawing("gjk1", d)

    # Second Iteration of GJK (the shapes above have been chosen to require this,
    # but in many cases the algorithm will terminate after the first iteration)
    # Compute the normal direction vector from the closest edge of the simplex
    p1, p2 = get_closest_edge(simplex[0], simplex[1], simplex[2])
    d4 = normal_direction_vector(p1, p2)
    if d4.dot(p1) > 0:
        d4 = -d4
    sp4 = support_function(a_verts, b_verts, d4)
    simplex = (p1, p2, sp4)

    d.clear()
    draw_axis(d)
    draw_polygons(d, a, b, md)
    draw_simplex(d, simplex)
    draw_normal(d, 1, d1, md_center + Vector2(2, 0), "cyan", a_verts, b_verts, True)
    draw_normal(d, 2, d2, sp1, "magenta", a_verts, b_verts, True)
    draw_normal(d, 3, d3, 0.5 * (sp1 + sp2), "orange", a_verts, b_verts, True)
    draw_normal(d, 4, d4, 0.5 * (p1 + p2), "blueviolet", a_verts, b_verts)
    save_drawing("gjk2", d)

    # voronoi
    d.clear()
    draw_axis(d)
    draw_voronoi(d, simplex)
    draw_polygons(d, a, b, None)
    save_drawing("gjk3", d)

    # GJK termination: test whether the origin is inside the simplex
    ca = simplex[0] - simplex[2]
    cb = simplex[1] - simplex[2]
    co = -simplex[2]
    RAC = triple_product(cb, ca, ca)
    RBC = triple_product(ca, cb, cb)
    if RAC.dot(co) > 0:
        # The origin is in RAC
        print("no collision")
    elif RBC.dot(co) > 0:
        # The origin is in RBC
        print("no collision")
    else:
        # The origin is in the triangle
        print("collision")


if __name__ == "__main__":
    main(save_svg)
