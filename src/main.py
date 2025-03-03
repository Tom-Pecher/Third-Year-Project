
from envs.default.default_env import DefaultTrafficEnv
from envs.random.random_env import RandomTrafficEnv
from envs.sanity.sanity_env import SanityTrafficEnv
from agents.DQN.DQN_Agent import DQNAgent

if __name__ == "__main__":
    default_env = DefaultTrafficEnv()
    random_env = RandomTrafficEnv()
    sanity_env = SanityTrafficEnv()

    agent1 = DQNAgent(random_env, wandb_on=False)
    agent1.train(100)

    agent2 = DQNAgent(sanity_env, wandb_on=False)
    agent2.load("DQN_100.pth")
    agent2.run(1, sumo_gui=True)
