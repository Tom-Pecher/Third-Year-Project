
# Replay memory allows the agent to learn from past experiences.

from collections import deque, namedtuple
import random
from utils.transition import Transition

# The replay memory class uses a deque to store the transitions:
class ReplayMemory:
    
    # Allocate a deque with a specified capacity:
    def __init__(self, capacity:int) -> None:
        self.memory = deque(maxlen=capacity)
        
    # Add a transition to the memory:
    def push(self, *args:tuple) -> None:
        self.memory.append(Transition(*args))
        
    # Sample a batch of transitions from the memory:
    def sample(self, batch_size:int) -> list:
        return random.sample(self.memory, batch_size)
    
    # Return the number of transitions stored in the memory:
    def __len__(self) -> int:
        return len(self.memory)
    