
import traci
import sumolib


class DefaultTrafficEnv():
    def __init__(self, config_path:str = "envs/default/sumo/env.sumocfg") -> None:
        self.config_path = config_path
        self.action_space = ((0, 1), (1, 0))
        # self.observation_space = [[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [0.0, 30.0]]
        self.observation_space = [[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]]
        self.last_switch = 0
        self.vehicle_waiting_times = {}

    def get_state(self) -> list:
        state = [traci.vehicle.getPosition(veh_id)[1] for veh_id in traci.vehicle.getIDList() if traci.vehicle.getPosition(veh_id)[0] > -5]
        if len(state) > 6:
            state = state[:6]
        else:
            while len(state) < 6:
                state.append(0.0)
        # state.append(float(max(traci.simulation.getTime() - self.last_switch, 10)))
        return state
        
    def reset(self, sumo_gui:bool=False) -> tuple:
        self.last_switch = 0
        self.vehicle_waiting_times = {}
        if traci.isLoaded():
            traci.load(["-c", self.config_path])
        else:
            self.sumoBinary = sumolib.checkBinary('sumo-gui' if sumo_gui else 'sumo')
            traci.start([self.sumoBinary, "-c", self.config_path, "--tripinfo-output", "envs/default/sumo_output/sumo_log.xml"])
        return self.get_state()
    
    def step(self, action:int) -> tuple:
        old_signal = traci.trafficlight.getRedYellowGreenState("J4")

        traci.simulationStep()
        if action == 1:
            traci.trafficlight.setRedYellowGreenState("J4", "rG")
        elif action == 0:
            traci.trafficlight.setRedYellowGreenState("J4", "Gr")
        else:
            raise ValueError("Invalid action")
        
        new_signal = traci.trafficlight.getRedYellowGreenState("J4")
        if new_signal != old_signal:
            self.last_switch = traci.simulation.getTime()
            # print(self.last_switch, new_signal)

        reward = 0

        reward -= sum((traci.vehicle.getWaitingTime(veh_id)**2 for veh_id in traci.vehicle.getIDList()))
        # print(reward)
            
        terminated = traci.simulation.getMinExpectedNumber() <= 0

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

        return state, reward, terminated, total_waiting_time
    
    def close(self) -> None:
        traci.close()
     
    
if __name__ == "__main__":
    pass
    