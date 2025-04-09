
import random
import sumolib
import traci

from envs.default import DefaultTrafficEnv
from utils.vehicle import Vehicle

# The random environment produces vehicles for random routes at random intervals:
class FixedFlowTrafficEnv(DefaultTrafficEnv):
    def __init__(self, simulation_name:str, state_type:int=0, reward_type:int=0, save_data:bool=False) -> None:
        super().__init__(simulation_name, state_type, reward_type, save_data)

    # Generate the route file for the vehicles:
    def generate_route_file(self) -> None:
        contents = ""
        with open(f"routes/{self.simulation_name}.rou.xml", "r") as f:
            contents = f.read()
        with open("sumo/env.rou.xml", "w") as f:
            f.write(contents)
