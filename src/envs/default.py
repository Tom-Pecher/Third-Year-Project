
# The default environment for the traffic simulation from which all other environments inherit.

import traci
import sumolib
import os
import xml.etree.ElementTree as ET
import math

from utils.vehicle import Vehicle

# The default environment produces vehicles for each possible route at 10 second intervals.
class DefaultTrafficEnv():
    def __init__(self, simulation_name:str, state_type:str="100", reward_type:str="100", save_data:bool=False) -> None:

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
    
    # Get distances of vehicles from junction:
    def get_distances(self) -> list:
        north = sorted([min(10000, traci.vehicle.getPosition(vehicle_id)[1]**2) for vehicle_id in traci.vehicle.getIDList() if "RoadNM" in traci.vehicle.getLaneID(vehicle_id)])
        south = sorted([min(10000, traci.vehicle.getPosition(vehicle_id)[1]**2) for vehicle_id in traci.vehicle.getIDList() if "RoadSM" in traci.vehicle.getLaneID(vehicle_id)])
        east = sorted([min(10000, traci.vehicle.getPosition(vehicle_id)[0]**2) for vehicle_id in traci.vehicle.getIDList() if "RoadEM" in traci.vehicle.getLaneID(vehicle_id)])
        west = sorted([min(10000, traci.vehicle.getPosition(vehicle_id)[0]**2) for vehicle_id in traci.vehicle.getIDList() if "RoadWM" in traci.vehicle.getLaneID(vehicle_id)])

        if len(north) < 5:
            north = north + [10000 for _ in range(5 - len(north))]
        if len(south) < 5:
            south = south + [10000 for _ in range(5 - len(south))]
        if len(east) < 5:
            east = east + [10000 for _ in range(5 - len(east))]
        if len(west) < 5:
            west = west + [10000 for _ in range(5 - len(west))]

        distances = north[:5] + south[:5] + east[:5] + west[:5]
        return distances
    
    # Get the current state of the simulation:
    def get_state(self, action=None, return_empty=False) -> tuple:
        if return_empty:
            length = 0
            if self.state_type[0] == "1":
                length += 2
            if self.state_type[1] == "1":
                length += len(traci.lanearea.getIDList())
            if self.state_type[2] == "1":
                length += 20
            return [0.0] * length
            
        else:
            action_time_diff = min(traci.simulation.getTime() - self.last_action_step, 50)
            if action is None:
                raise Exception("Action is None")
            action = float(action)

        state = []
        if self.state_type[0] == "1":
            state.append(action)
            state.append(action_time_diff)
        if self.state_type[1] == "1":
            for queue in self.get_queues():
                state.append(queue)
        if self.state_type[2] == "1":
            for distance in self.get_distances():
                state.append(distance)
        return state


    
    # Get the current reward of the simulation:
    def get_reward(self) -> float:
        reward = 0
        if self.state_type[0] == "1":
            t_delta = traci.simulation.getTime() - self.last_action_step
            switch_point = 25;
            max_reward = 5;
            gradient = 1;
            if self.state_changed:
                time_penalty = min(gradient * (t_delta - switch_point), max_reward)
            else:
                time_penalty = min(-gradient * (t_delta - switch_point), max_reward)
            reward += time_penalty
        if self.state_type[1] == "1":
            reward -= sum(self.get_queues())
        if self.state_type[2] == "1":
            reward -= sum([vehicle.waiting_time for vehicle in self.vehicles if vehicle.in_simulation])
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
