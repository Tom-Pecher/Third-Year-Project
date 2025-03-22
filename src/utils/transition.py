
# In this project, transitions represent the experience of the agent after taking an action in a state.

# Named tuple are used since the values are immutable (faster and more memory efficient):
from collections import namedtuple

# The main data structure for storing transitions is a named tuple containing the following values:
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
