from rigidphysics.config import Resolution
from rigidphysics.flat import Simulation


simulation = Simulation(Resolution(800, 600), debug=True, show_contacts=False)
simulation.run()
