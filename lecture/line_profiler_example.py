import math
import random
from typing import List, NamedTuple, Tuple
import line_profiler


Circle = NamedTuple("Circle", [("x", float), ("y", float), ("radius", float)])


def random_circle():
    x = random.uniform(-30, 30)
    y = random.uniform(-30, 30)
    radius = random.uniform(1.5, 2.5)
    return Circle(x, y, radius)


@line_profiler.profile
def find_intersections(circles: List[Circle]) -> List[Tuple[int, int]]:
    num_circles = len(circles)
    intersections = []
    for i in range(num_circles):
        x1, y1, r1 = circles[i]
        for j in range(0, i):
            x2, y2, r2 = circles[j]
            dx = x2 - x1
            dy = y2 - y1
            distance_sq = dx * dx + dy * dy
            radius_sum_sq = (r1 + r2)**2
            if distance_sq < radius_sum_sq:
                intersections.append((i, j))

    return intersections


def main():
    num_circles = 1000
    circles = [random_circle() for _ in range(num_circles)]
    intersections = find_intersections(circles)
    print(f"Number of intersections: {len(intersections)}")


if __name__ == "__main__":
    main()
