
from envs.test_env import TestEnv
from agents.DQN_Agent import DQNAgent

if __name__ == "__main__":
    env = TestEnv()
    agent1 = DQNAgent(env)
    agent1.train(300)
    agent1.save("DQN_300.pth")

    agent2 = DQNAgent(env)
    agent2.load("DQN_300.pth")
    agent2.run()
