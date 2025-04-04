
import random
import sumolib
import traci
import xml.etree.ElementTree as ET

from envs.default import DefaultTrafficEnv
from utils.vehicle import Vehicle

# The random environment produces vehicles for random routes at random intervals:
class SanityTrafficEnv(DefaultTrafficEnv):
    def __init__(self, simulation_name:str, save_data:bool=False) -> None:
        super().__init__(simulation_name, save_data)

    # Generate the route file for the vehicles:
    def generate_route_file(self) -> None:
        tree = ET.parse(f'routes/{self.simulation_name}.rou.xml')
        root = tree.getroot()

        route = ""
        routes = []
        for flow in root.findall('flow'):
            route = flow.get('from')
            route += " "
            if flow.get('via') is not None:
                route += " ".join(flow.get('via'))
            route += " "
            route += flow.get('to')
            routes.append(route)

        with open("sumo/env.rou.xml", "w") as f:
            f.write('<routes>\n')
            for i in range(len(routes)):
                f.write(f'\t<route id="route_{i}" edges="{routes[i]}"/>\n')

            v = 0
            for r in range(len(routes)):
                for _ in range(10):
                    f.write(f'<vehicle id="{v}" route="route_{r}" depart="{10*v + 100*r}"/>\n')
                    v += 1

            start_overflow = 10*v + 100*len(routes) + 100

            for d in range(40):
                for r in range(len(routes)):
                    f.write(f'<vehicle id="{v}" route="route_{r}" depart="{5*d + start_overflow}"/>\n')
                    v += 1
            f.write('</routes>')
