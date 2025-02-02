
from envs.basic_random.basic_random_env import BasicRandomTrafficEnv
from agents.DQN.DQN_Agent import DQNAgent

if __name__ == "__main__":
    env = BasicRandomTrafficEnv("envs/basic_random/sumo/basic_random.sumocfg")

    agent1 = DQNAgent(env)
    agent1.train(100)

    agent2 = DQNAgent(env)
    agent2.load("DQN_100.pth")
    agent2.run(1, sumo_gui=True)
