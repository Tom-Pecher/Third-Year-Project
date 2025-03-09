
import traci
import sumolib
import numpy as np

from envs.default.default_env import DefaultTrafficEnv

class RandomTrafficEnv(DefaultTrafficEnv):
    def __init__(self, config_path:str = "envs/sumo/env.sumocfg", state_type=0, vehicles_seen=6) -> None:
        super().__init__(config_path, state_type, vehicles_seen)

    def generate_routefile(self) -> None:
        with open("envs/sumo/env.rou.xml", "w") as routes:
            routes.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Define routes -->
    <route id="route_0" edges="E5 E1"/>
    <route id="route_1" edges="E6 E1"/>

    <!-- Define vehicles that use the routes -->\n""")
            departures = sorted(np.random.uniform(0, 1000, 200))
            for i in range(len(departures)):
                routes.write(f"""<vehicle id="{i}" route="route_{np.random.choice((0, 1))}" depart="{departures[i]}"/>\n""")
            
            routes.write("</routes>")
        
    def reset(self, sumo_gui:bool=False) -> tuple:
        self.generate_routefile()
        return super().reset(sumo_gui)
   
if __name__ == "__main__":
    pass
    