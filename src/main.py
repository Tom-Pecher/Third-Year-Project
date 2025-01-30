
from envs.basic.basic_env import BasicTrafficEnv
from agents.Traffic_DQN_Agent import DQNAgent

if __name__ == "__main__":
    env = BasicTrafficEnv("envs/basic/sumo/basic.sumocfg")

    # agent1 = DQNAgent(env)
    # agent1.train(300)
    # agent1.save("DQN_300.pth")

    agent2 = DQNAgent(env)
    agent2.load("DQN_300.pth")
    agent2.run(1, sumo_gui=True)
