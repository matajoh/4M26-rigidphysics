"""Script that draws a quadtree onto a snapshot of the simulation.

Out of scope for the Tripos.
"""

from colorsys import hls_to_rgb
import json
import math
import random
from typing import NamedTuple
from PIL import Image, ImageDraw

from rigidphysics.flat.bodies import AABB
from rigidphysics.flat.quadtree import Node, QuadTree


DummyBody = NamedTuple("DummyBody", [("aabb", AABB)])


def load_aabbs():
    with open("snapshot.json") as file:
        bodies = json.load(file)

    aabbs = []
    for b in bodies:
        if b["kind"] == "circle":
            x, y = b["position"]
            r = b["radius"]
            aabbs.append(AABB(x - r, y - r, x + r, y + r))
        else:
            xmin, ymin = math.inf, math.inf
            xmax, ymax = -math.inf, -math.inf
            px, py = b["position"]
            cos = math.cos(b["angle"])
            sin = math.sin(b["angle"])
            for x, y in b["vertices"]:
                xt = px + cos * x - sin * y
                yt = py + sin * x + cos * y
                xmin = min(xt, xmin)
                ymin = min(yt, ymin)
                xmax = max(xt, xmax)
                ymax = max(yt, ymax)

            aabbs.append(AABB(xmin, ymin, xmax, ymax))

    return aabbs


def draw_node(node: Node, draw: ImageDraw, cx: float, cy: float, scale: float):
    hue = random.uniform(0, 1)
    r, g, b = hls_to_rgb(hue, 0.5, 1.0)
    color = int(r * 255), int(g * 255), int(b * 255)
    box = node.box
    center = box.center
    draw.line([cx + box.left * scale, cy + center.y * scale,
               cx + box.right * scale, cy + center.y * scale], fill="gray", width=4)
    draw.line([cx + center.x * scale, cy + box.top * scale,
               cx + center.x * scale, cy + box.bottom * scale], fill="gray", width=4)

    for body in node.values:
        aabb = body.aabb
        left = cx + aabb.left * scale
        top = cy + aabb.top * scale
        right = cx + aabb.right * scale
        bottom = cy + aabb.bottom * scale
        draw.rectangle([left, top, right, bottom], outline=color, width=2)

    if node.is_leaf:
        return

    for child in node:
        draw_node(child, draw, cx, cy, scale)


def main():
    random.seed(0)
    snapshot = Image.open("snapshot.png")
    snapshot = snapshot.convert("L")
    image = Image.new("RGB", snapshot.size, "white")
    aabbs = load_aabbs()
    scale = image.height / 30
    width = image.width / scale
    height = 30
    quadtree = QuadTree(AABB(-width, -height, width, height))
    for aabb in aabbs:
        quadtree.add(DummyBody(aabb))

    cx, cy = image.width / 2, image.height / 2
    draw = ImageDraw.Draw(image)
    image.paste(snapshot, (0, 0))
    draw_node(quadtree.root, draw, cx, cy, scale)

    image.save("quadtree.png")


if __name__ == "__main__":
    main()
