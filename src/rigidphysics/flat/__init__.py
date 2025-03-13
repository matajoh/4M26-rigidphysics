"""Flat (2D) Rigid Physics Simulation.

These files, in addition with the README, act as lecture notes and a
revision aid for the Tripos. Some files are not in scope for the Tripos
and will be marked as such in the module comments. Other files will have
commented methods or functions which correspond to those discussed in
lecture, and students should understand those thoroughly.

This version of the physics simulation provides an unoptimised, easier
to understand version of the system.  You may notice that many of the
algorithms are slightly modified versions of those we discussed
in lecture such that they can accommodate any convex polygon.
"""

from argparse import ArgumentParser

from ..config import DetectionKind, Resolution
from .engine import PhysicsMode
from .simulation import Simulation


def main():
    parser = ArgumentParser(description="2D Rigid Physics Simulation")
    parser.add_argument("--mode", "-m", help="Physics mode to use",
                        choices=["basic", "friction", "none", "rotation"],
                        default="friction")
    parser.add_argument("--show-contacts", action="store_true", help="Show contact points")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--snapshot", "-ss", action="store_true", help="Save a snapshot of the simulation")
    parser.add_argument("--size", "-s", help="Size of the window", default="1200x900")
    parser.add_argument("--detect", type=str, help="Detection algorithm to use",
                        choices=["quadtree", "basic"], default="quadtree")
    args = parser.parse_args()

    simulation = Simulation(resolution=Resolution.from_string(args.size),
                            physics_mode=PhysicsMode[args.mode.upper()],
                            detection_kind=DetectionKind[args.detect.upper()],
                            debug=args.debug,
                            show_contacts=args.show_contacts,
                            snapshot=args.snapshot)
    simulation.run()
