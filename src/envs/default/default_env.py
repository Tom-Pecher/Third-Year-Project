
import traci
import sumolib
import sys
import io

class DefaultTrafficEnv():
    def __init__(self, config_path:str = "envs/default/sumo/env.sumocfg") -> None:
        self.config_path = config_path
        self.action_space = ((0, 1), (1, 0))
        # self.observation_space = [[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [0.0, 30.0]]
        self.observation_space = [[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]]
        self.last_switch = 0
        self.vehicle_waiting_times = {}
        self.emergency_brakes = 0

        self.queue_lengths = {'E5': [], 'E6': []} 

        self.stderr_capture = io.StringIO()
        sys.stderr = self.stderr_capture

        self.time_loss = 0

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
        self.time_loss = 0
        self.emergency_brakes = 0
        self.last_switch = 0
        self.vehicle_waiting_times = {}
        self.queue_lengths = {'E5': [], 'E6': []}

        self.stderr_capture.truncate(0)
        self.stderr_capture.seek(0)

        if traci.isLoaded():
            traci.load(["-c", self.config_path])
        else:
            self.sumoBinary = sumolib.checkBinary('sumo-gui' if sumo_gui else 'sumo')
            traci.start([self.sumoBinary, "-c", self.config_path, "--tripinfo-output", "envs/default/sumo_output/sumo_log.xml"])
        return self.get_state()
    
    def step(self, action:int) -> tuple:
        old_signal = traci.trafficlight.getRedYellowGreenState("J4")

        traci.simulationStep()

        self.queue_lengths['E5'].append(self.get_queue_length('E5'))
        self.queue_lengths['E6'].append(self.get_queue_length('E6'))

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

        self.check_emergency_brakes()
        self.get_time_loss()
        avg_queues = self.get_average_queue_lengths()

        episode_info = {
            'total_waiting_time': sum(self.vehicle_waiting_times.values()),
            'avg_queue_E5': avg_queues['E5'],
            'avg_queue_E6': avg_queues['E6'],
            'emergency_brakes': self.emergency_brakes,
            'time_loss': self.time_loss
        }

        return state, reward, terminated, episode_info
    
    def close(self) -> None:
        sys.stderr = sys.__stderr__
        traci.close()

    def get_queue_length(self, edge_id:str) -> int:
        """Calculate number of vehicles waiting (speed < 0.1 m/s) on an edge"""
        queue_length = 0
        for veh_id in traci.edge.getLastStepVehicleIDs(edge_id):
            if traci.vehicle.getSpeed(veh_id) < 0.1:
                queue_length += 1
        return queue_length
    
    def get_average_queue_lengths(self) -> dict:
        """Calculate average queue lengths for the entire episode"""
        return {
            'E5': sum(self.queue_lengths['E5']) / len(self.queue_lengths['E5']) if self.queue_lengths['E5'] else 0,
            'E6': sum(self.queue_lengths['E6']) / len(self.queue_lengths['E6']) if self.queue_lengths['E6'] else 0
        }
    
    def check_emergency_brakes(self) -> int:
        """Check for emergency braking events"""
        # Occasionally detects false positives (replace with better approach)
        for veh_id in traci.vehicle.getIDList():
            acceleration = traci.vehicle.getAcceleration(veh_id)
            if acceleration <= -8:  # Threshold for emergency braking (m/s²)
                self.emergency_brakes += 1

    def get_time_loss(self) -> float:
        self.time_loss += sum([traci.vehicle.getTimeLoss(veh_id) for veh_id in traci.vehicle.getIDList()])
    
    
if __name__ == "__main__":
    pass
    