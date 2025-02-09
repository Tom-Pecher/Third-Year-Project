
import traci
import sumolib


class BasicSanityTrafficEnv():
    def __init__(self, config_path:str = "sumo/basic_sanity.sumocfg") -> None:
        self.config_path = config_path
        self.action_space = ((0, 1), (1, 0))
        self.observation_space = [[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [0.0, 30.0]]
        self.time_step = 0
        self.last_switch = 0
        self.vehicle_waiting_times = {}

    def get_state(self) -> list:
        state = [traci.vehicle.getPosition(veh_id)[1] for veh_id in traci.vehicle.getIDList() if traci.vehicle.getPosition(veh_id)[0] > -5]
        if len(state) > 5:
            state = state[:5]
        else:
            while len(state) < 5:
                state.append(0.0)
        state.append(float(max(self.time_step - self.last_switch, 10)))
        return state
        
    def reset(self, sumo_gui:bool=False) -> tuple:
        self.time_step = 0
        self.last_switch = 0
        self.vehicle_waiting_times = {}
        if traci.isLoaded():
            traci.load(["-c", self.config_path])
        else:
            self.sumoBinary = sumolib.checkBinary('sumo-gui' if sumo_gui else 'sumo')
            traci.start([self.sumoBinary, "-c", self.config_path, "--tripinfo-output", "envs/basic_sanity/sumo_output/sumo_log.xml"])
        return self.get_state()
    
    def step(self, action:int) -> tuple:
        self.time_step += 1

        traci.simulationStep()
        if action == 1 and traci.trafficlight.getRedYellowGreenState("J4") != "rG":
            traci.trafficlight.setRedYellowGreenState("J4", "rG")
            self.last_switch = self.time_step
        elif action == 0 and traci.trafficlight.getRedYellowGreenState("J4") != "Gr":
            traci.trafficlight.setRedYellowGreenState("J4", "Gr")
            self.last_switch = self.time_step
        else:
            pass
            # raise ValueError("Invalid action")
        terminated = traci.simulation.getMinExpectedNumber() <= 0

        reward = 0
        state = self.get_state()
        if terminated:
            reward += 1000

        for veh_id in traci.vehicle.getIDList():
            waiting_time = traci.vehicle.getWaitingTime(veh_id)
            if veh_id not in self.vehicle_waiting_times:
                self.vehicle_waiting_times[veh_id] = waiting_time
            else:
                if waiting_time > self.vehicle_waiting_times[veh_id]:
                    self.vehicle_waiting_times[veh_id] = waiting_time

        total_waiting_time = sum(self.vehicle_waiting_times.values())

        reward -= sum((traci.vehicle.getWaitingTime(veh_id) for veh_id in traci.vehicle.getIDList()))
        if self.time_step - self.last_switch < 8:
            reward -= 100

        # reward -= total_waiting_time

        return state, reward, terminated, total_waiting_time
    
    def close(self) -> None:
        traci.close()
     
    
if __name__ == "__main__":
    pass
    