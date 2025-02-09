
from envs.basic_sanity.basic_sanity_env import BasicSanityTrafficEnv
from agents.DQN.DQN_Agent import DQNAgent

if __name__ == "__main__":
    env = BasicSanityTrafficEnv("envs/basic_sanity/sumo/basic_sanity.sumocfg")

    agent1 = DQNAgent(env, wandb_on=True)
    agent1.train(1000)

    agent2 = DQNAgent(env)
    agent2.load("DQN_1000.pth")
    agent2.run(1, sumo_gui=True)
