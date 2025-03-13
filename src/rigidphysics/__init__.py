"""Rigid Physics Simulation.

These files, in addition with the README, act as lecture notes and a
revision aid for the Tripos. Some files are not in scope for the Tripos
and will be marked as such in the module comments. Other files will have
commented methods or functions which correspond to those discussed in
lecture, and students should understand those thoroughly.

This version of the physics simulation is fairly optimised and near to
what might be a final version (of a system of this kind). It utilises
both vectorised compute (via numpy) and numba for JIT compilation. To
see a simpler version of the simulation, please look at the flat submodule.
"""

from argparse import ArgumentParser

from .config import (
    Resolution,
    SimulationConfig,
)
from .simulation import Simulation


def parse_args():
    parser = ArgumentParser(description="Render a simulation")
    parser.add_argument("--config", "-c", help="Path to the simulation config file")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--show-contacts", action="store_true", help="Show contact points")
    parser.add_argument("--mode", "-i", help="Physics mode",
                        choices=["basic", "rotation", "friction"],
                        default="friction")
    parser.add_argument("--resolution", "-r", help="Resolution of the simulation",
                        default="800x600")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        config = SimulationConfig.load(args.config)._replace(
            resolution=Resolution.from_string(args.resolution),
            debug=args.debug,
            show_contacts=args.show_contacts)
    else:
        config = SimulationConfig.create(debug=args.debug,
                                         show_contacts=args.show_contacts,
                                         mode=args.mode,
                                         resolution=Resolution.from_string(args.resolution))

    simulation = Simulation(config)
    simulation.run()
