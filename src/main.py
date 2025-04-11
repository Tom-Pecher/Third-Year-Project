
import os
import sys

from agents.default import DefaultAgent
from agents.fixed_duration import FixedDurationAgent
from agents.random import RandomAgent
from agents.dqn import DQNAgent
from agents.ddqn import DDQNAgent
from agents.dddqn import DDDQNAgent

from agents.test_dqn import Test_DQNAgent

from envs.default import DefaultTrafficEnv
from envs.random import RandomTrafficEnv
from envs.sanity import SanityTrafficEnv

if __name__ == "__main__":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")

    # r = RandomTrafficEnv("X_3", state_type=3, reward_type=0)
    # r = SanityTrafficEnv("X_3", state_type=2)

    # a = Test_DQNAgent(r)
    # a = DQNAgent(r, wandb_on=False)
    # a = DDQNAgent(r, wandb_on=False)
    # a = DDDQNAgent(r, wandb_on=True)
    # a = FixedDurationAgent(r, 100, wandb_on=False)
    # a = RandomAgent(r, 0.5, wandb_on=False)

    # a.train(1, sumo_gui=False)
    # a.run(1, sumo_gui=True)

    # a.load("DDDQN_20.pth")
    # a.train(1, sumo_gui=True)
        
    r = RandomTrafficEnv("X_3")
    a = None

    for i in range(0, 61, 5):
        a = FixedDurationAgent(r, i, wandb_on=False)
        a.run(1, sumo_gui=False, id=i)
