
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

    # Choose an environment:
    env = RandomTrafficEnv("X_3", state_type="101", reward_type="110")
    # Select:
    #  - a road network (e.g. X_3)
    #  - state type (e.g. 101)
    #  - reward type (e.g. 110)

    # Choose an agent:
    agent = DDDQNAgent(env, wandb_on=False)

    # Train the agent:
    agent.train(100)

    # Run the agent and view its performance:
    agent.run(1, sumo_gui=True)
