
import traci
import sumolib
import numpy as np


class BasicRandomTrafficEnv():
    def __init__(self, config_path:str = "sumo/basic.sumocfg") -> None:
        self.config_path = config_path
        self.action_space = ((0, 1), (1, 0))
        self.observation_space = [[50.0, 0.0], [-50.0, 0.0]]

    def generate_routefile(self) -> None:
        with open("envs/basic_random/sumo/basic_random.rou.xml", "w") as routes:
            routes.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Define routes -->
    <route id="route_0" edges="E5 E1"/>
    <route id="route_1" edges="E6 E1"/>

    <!-- Define vehicles that use the routes -->\n""")
            departures = sorted(np.random.uniform(0, 100, 100))
            for i in range(100):
                routes.write(f"""<vehicle id="{i}" route="route_{np.random.choice((0, 1))}" depart="{departures[i]}"/>\n""")
            
            routes.write("</routes>")

    def get_state(self) -> list:
        state = [traci.vehicle.getPosition(veh_id)[1] for veh_id in traci.vehicle.getIDList() if traci.vehicle.getPosition(veh_id)[0] > -5]
        if len(state) > 2:
            state = state[:2]
        else:
            while len(state) < 2:
                state.append(0.0)
        return state
        
    def reset(self, sumo_gui:bool=False) -> tuple:
        self.generate_routefile()
        if traci.isLoaded():
            traci.load(["-c", self.config_path])
        else:
            self.sumoBinary = sumolib.checkBinary('sumo-gui' if sumo_gui else 'sumo')
            traci.start([self.sumoBinary, "-c", self.config_path, "--tripinfo-output", "envs/basic_random/sumo_output/sumo_log.xml"])
        return self.get_state()
    
    def step(self, action:int) -> tuple:
        traci.simulationStep()
        if action == 1:
            traci.trafficlight.setRedYellowGreenState("J4", "rG")
        elif action == 0:
            traci.trafficlight.setRedYellowGreenState("J4", "Gr")
        else:
            raise ValueError("Invalid action")

        reward = -sum((traci.vehicle.getWaitingTime(veh_id) for veh_id in traci.vehicle.getIDList()))
        terminated = traci.simulation.getMinExpectedNumber() <= 0

        state = self.get_state()
        if terminated:
            reward += 1000

        return state, reward, terminated
    
    def close(self) -> None:
        traci.close()
     
    
if __name__ == "__main__":
    pass
    