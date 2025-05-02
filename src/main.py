
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
from envs.sanity import SanityTrafficEnv

if __name__ == "__main__":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")

    # r = RandomTrafficEnv("X_3", state_type="111", reward_type="111")
    # r = SanityTrafficEnv("X_3", state_type=2)

    # a = DQNAgent(r, wandb_on=False)
    # a = DDQNAgent(r, wandb_on=False)
    # a = DDDQNAgent(r, wandb_on=True)
    # a = FixedDurationAgent(r, 100, wandb_on=False)
    # a = RandomAgent(r, 0.5, wandb_on=False)

    # a.train(1, sumo_gui=False)
    # a.run(1, sumo_gui=True)

    # a.load("DDDQN_20.pth")
    # a.train(1, sumo_gui=True)
        
    env = RandomTrafficEnv("X_3", state_type="111", reward_type="011")

    agent = DDDQNAgent(env, wandb_on=False)
    agent.train(25)
    agent.run(1, sumo_gui=True)

    # env2 = RandomTrafficEnv("T_1", state_type="111", reward_type="010")

    # dddqn_agent = DQNAgent(env2, wandb_on=False)
    # dddqn_agent.load("DQN_25.pth")
    # dddqn_agent.run(1, sumo_gui=True)