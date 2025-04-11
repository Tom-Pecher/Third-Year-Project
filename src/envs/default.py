
# The default environment for the traffic simulation from which all other environments inherit.

import traci
import sumolib
import os
import xml.etree.ElementTree as ET
import math

from utils.vehicle import Vehicle

# The default environment produces vehicles for each possible route at 10 second intervals.
class DefaultTrafficEnv():
    def __init__(self, simulation_name:str, state_type:int=0, reward_type:int=0, save_data:bool=False) -> None:

        self.simulation_name = simulation_name
        self.state_type = state_type
        self.reward_type = reward_type

        # Obtain the path to the src directory:
        self.src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Obtain the paths to the configuration files:
        self.config_path = os.path.join(self.src_dir, "sumo/env.sumocfg")
        self.network_path = os.path.join(self.src_dir, f"networks/{simulation_name}.net.xml")
        self.sensor_path = os.path.join(self.src_dir, f"sensors/{simulation_name}.add.xml")
        self.routes_path = os.path.join(self.src_dir, f"routes/{simulation_name}.rou.xml")
        self.save_data = save_data

        # Load the network file and obtain the light phases and vehicle routes:
        self.load_network()
        self.vehicles = []

        # Configure data saving:
        if self.save_data:
            self.version = 0
            self.data_filename = f"saved/data/{self.network_path.split('/')[-1].split('.')[0]}({self.version}).csv"
            while os.path.exists(self.data_filename):
                self.version += 1
                self.data_filename = f"saved/data/{self.network_path.split('/')[-1].split('.')[0]}({self.version}).csv"
            with open(self.data_filename, "x") as f:
                f.write(",".join(self.get_episode_info().keys()) + '\n')

        self.last_action_step = 0
        self.action_step = 0
        self.state_changed = False
        
    # Load the desired road network into the SUMO directory:
    def load_network(self) -> None:
        # Clear the SUMO directory:
        keep_names = ('env.sumocfg', 'env.net.xml', 'env.rou.xml', 'env.add.xml', 'test.py')
        for filename in os.listdir('sumo'):
            file_path = os.path.join('sumo', filename)
            if os.path.isfile(file_path) and filename not in keep_names:
                os.remove(file_path)

        # Load the network file into the SUMO directory:
        with open(self.network_path, "r") as f:
            network_config = f.read()
            with open("sumo/env.net.xml", "w") as f:
                f.write(network_config)

        # If the network has sensors, load them into the SUMO directory:
        if self.sensor_path is not None:
            with open(self.sensor_path, "r") as f:
                sensor_config = f.read()
                with open("sumo/env.add.xml", "w") as f:
                    f.write(sensor_config)
        else:
            with open(self.sensor_path, "w") as f:
                f.write('<additional></additional>')

        self.generate_route_file()

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
            for j in range(len(routes)):
                f.write(f'<vehicle id="vehicle_{j}" route="route_{j}" depart="{j*10}"/>\n')
            f.write('</routes>')


    # Reset/start the simulation:
    def reset(self, sumo_gui:bool=False) -> tuple:
        self.vehicles = []
        self.last_action_step = 0
        self.action_step = 0
        
        if traci.isLoaded():
            traci.load(["-c", self.config_path])
        else:
            self.sumoBinary = sumolib.checkBinary('sumo-gui' if sumo_gui else 'sumo')
            traci.start([self.sumoBinary, "-c", self.config_path])

        logics = [phase for phase in traci.trafficlight.getAllProgramLogics("TCS")]
        phases = [phase.phases for phase in logics][0]
        self.phases = [phase.state for phase in phases]

        # self.observation_space = [0.0 for _ in traci.lanearea.getIDList()] + [0.0] + [0.0]
        self.observation_space = self.get_state(return_empty=True)
        self.action_space = range(len(self.phases))

        return self.get_state(return_empty=True)
    
    # Advance the simulation by one step:
    def step(self, action:int, yellows:bool=True) -> tuple:
        self.state_changed = False
    
        if action is not None:
            if type(action) is not int:
                action = int(action)
            current_phase = traci.trafficlight.getRedYellowGreenState("TCS")
            if self.phases[action] != current_phase:
                print(f"CHANGING PHASE from {current_phase} to {self.phases[action]}")
                self.state_changed = True
                self.last_action_step = self.action_step
                self.action_step = traci.simulation.getTime()
                traci.trafficlight.setRedYellowGreenState("TCS", self.phases[action])

        traci.simulationStep()
        
        state = self.get_state(action=action)
        reward = self.get_reward()
        terminated = traci.simulation.getMinExpectedNumber() <= 0

        print(f"Time:{traci.simulation.getTime()}, State: {state}, Action: {action}, Reward: {reward}")

        self.update_vehicles()
        env_info = self.get_env_info()
        episode_info = self.get_episode_info()

        if terminated:
            if self.save_data:
                with open(self.data_filename, "a") as f:
                    f.write(",".join(map(str, episode_info.values())) + '\n')

        # print(self.get_queues())

        return state, reward, terminated, env_info
    
    # Get queue lengths at each detector:
    def get_queues(self) -> list:
        return [traci.lanearea.getLastStepVehicleNumber(detector_id) for detector_id in traci.lanearea.getIDList()]
    
    # Get the current state of the simulation:
    def get_state(self, action=None, return_empty=False) -> tuple:
        # queues = [min(queue, 5) for queue in self.get_queues()]
        
        # current_phase = self.phases.index(traci.trafficlight.getRedYellowGreenState("TCS"))
        # return queues + [action_time_diff] + [current_phase]

        if return_empty:
            action_time_diff = 0.0
            action = 0.0
        else:
            action_time_diff = min(traci.simulation.getTime() - self.last_action_step, 50)
            if action is None:
                raise Exception("Action is None")
            action = float(action)

        match self.state_type:
            case 0:
                return [action_time_diff]
            case 1:
                return [action, action_time_diff]
            case 2:
                return [action, action_time_diff] + self.get_queues()
            case 3:
                return self.get_queues()
            case _:
                raise Exception("Invalid state type")

    
    # Get the current reward of the simulation:
    def get_reward(self) -> float:
        # queues = [min(queue**2, 50) for queue in self.get_queues()]
        waiting_times = [min(vehicle.waiting_time, 10) for vehicle in self.vehicles if vehicle.in_simulation]

        action_time_diff = traci.simulation.getTime() - self.last_action_step
        time_last_action = 0
        # print(traci.simulation.getTime(), self.action_step, self.last_action_step)
        DELAY = 13
        C = 50
        DROPOFF = 2.5
        if self.state_changed:
            # time_last_action = C if action_time_diff < DELAY else min(C/(action_time_diff - DELAY + 1), C)
            time_last_action = min(C, max(0, C - DROPOFF * (action_time_diff - DELAY)))

        # reward = sum(waiting_times) + 2*time_last_action**1.5

        match self.reward_type:
            case 0:
                reward = -sum(waiting_times)
            case 1:
                reward = -2*time_last_action**1.5
            case 2:
                reward = -(sum(waiting_times) + 2*time_last_action**1.5)
            case 3:
                reward = 20*action_time_diff**2 / ((action_time_diff - 40)**2 + 100) - 200 if action_time_diff < 40 else -1000
            case 4:
                if self.state_changed:
                    reward = -100 if action_time_diff < 30 else 400
                    # print("STATE CHANGED", action_time_diff, reward)
                else:
                    reward = 0 if action_time_diff < 30 else -500
                    # print("STATE NOT CHANGED", time_last_action, reward)
            case 5:
                if self.state_changed:
                    reward = -100 * math.tanh((action_time_diff - 40)/10)
                else:
                    reward = -(action_time_diff - 20) * (action_time_diff + 20) / 4
            case _:
                raise Exception("Invalid state type")

        return reward

    # Update the vehicles in the simulation:
    def update_vehicles(self) -> None:
        for v in traci.vehicle.getIDList():
            if v not in [veh.id for veh in self.vehicles]:
                vehicle = Vehicle(v)
                self.vehicles.append(vehicle)
        
        for veh in self.vehicles:
            if veh.in_simulation:
                if veh.id not in traci.vehicle.getIDList():
                    veh.in_simulation = False
                else:
                    veh.update()

    # Get environment information:
    def get_env_info(self) -> dict:
        active_vehicles = [vehicle for vehicle in self.vehicles if vehicle.in_simulation]
        return {
            "active_vehicles"    : len(active_vehicles),
            "throughput"         : len(traci.simulation.getArrivedIDList()),
            "gross_waiting_time" : sum([vehicle.waiting_time for vehicle in active_vehicles]),
            "n_severe_brakes" : sum([vehicle.severe_brakes for vehicle in active_vehicles]),
            "gross_time_loss"    : sum([vehicle.time_loss for vehicle in active_vehicles]),
        }
    
    # Get environment information over full episode:
    def get_episode_info(self) -> dict:
        return {
            "cumulative_waiting_time": sum([vehicle.cumulative_waiting_time for vehicle in self.vehicles]),
            "n_severe_brakes": sum([vehicle.severe_brakes for vehicle in self.vehicles]),
            "cumulative_time_loss": sum([vehicle.cumulative_time_loss for vehicle in self.vehicles]),
        }


    # Close the simulation:
    def close(self) -> None:
        traci.close()
