
from envs.test_env import TestEnv
from agents.DQN import DQNAgent

if __name__ == "__main__":
    env = TestEnv()
    agent = DQNAgent(env)
    agent.train()
    