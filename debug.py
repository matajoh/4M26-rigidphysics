from rigidphysics import Simulation, SimulationConfig


config = SimulationConfig.create(debug=True, show_contacts=False, max_instances=100)
simulation = Simulation(config)
simulation.run()
