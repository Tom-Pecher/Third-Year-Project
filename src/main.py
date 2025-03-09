
from envs.default.default_env import DefaultTrafficEnv
from envs.random.random_env import RandomTrafficEnv
from envs.sanity.sanity_env import SanityTrafficEnv

from agents.DQN.DQN_Agent import DQNAgent
from agents.DDQN.DDQN_Agent import DDQNAgent
from agents.DDDQN.DDDQN_Agent import DDDQNAgent

if __name__ == "__main__":
    default_env = DefaultTrafficEnv()
    random_env = RandomTrafficEnv(state_type=1, vehicles_seen=10)
    sanity_env = SanityTrafficEnv()

    agent1 = DDDQNAgent(random_env, wandb_on=False)
    agent1.train(1000)

    agent2 = DDDQNAgent(sanity_env, wandb_on=False)
    agent2.load("DQN_1000.pth")
    agent2.run(1, sumo_gui=True)
