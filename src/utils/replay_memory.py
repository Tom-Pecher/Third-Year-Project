
import random
from collections import namedtuple, deque
from utils.transition import Transition

class ReplayMemory:
    def __init__(self, capacity:int) -> None:
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args:tuple) -> None:
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size:int) -> list:
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)