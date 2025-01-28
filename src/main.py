
from envs.test_env import TestEnv
from agents.DQN_Agent import DQNAgent

if __name__ == "__main__":
    env = TestEnv()
    agent = DQNAgent(env)
    agent.train()
