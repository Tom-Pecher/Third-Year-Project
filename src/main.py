
import os
import sys

from agents.default import DefaultAgent
from agents.fixed_duration import FixedDurationAgent
from agents.random import RandomAgent
from agents.dqn import DQNAgent
from agents.ddqn import DDQNAgent
from agents.dddqn import DDDQNAgent

from envs.default import DefaultTrafficEnv
from envs.random import RandomTrafficEnv

if __name__ == "__main__":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")


    r = RandomTrafficEnv("X_3")
    # a = DQNAgent(r, wandb_on=True)
    # a = DDQNAgent(r, wandb_on=True)
    a = DDDQNAgent(r, wandb_on=True)
    # a = FixedDurationAgent(r, 100, wandb_on=True)
    # a = RandomAgent(r, 0.5, wandb_on=True)
    a.train(0, sumo_gui=False)
    # a.run(1, sumo_gui=True)
