
import gym

class TestEnv():
    def __init__(self, env_name:str="CartPole-v1") -> None:
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def reset(self) -> tuple:
        return self.env.reset()
    
    def step(self, action:int) -> tuple:
        return self.env.step(action)
    