
import traci
import sumolib

from envs.default.default_env import DefaultTrafficEnv

class SanityTrafficEnv(DefaultTrafficEnv):
    def __init__(self, config_path:str = "envs/sumo/env.sumocfg") -> None:
        super().__init__(config_path)
    
if __name__ == "__main__":
    pass
    