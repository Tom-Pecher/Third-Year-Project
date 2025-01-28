
# This iteration of DQN is a simple initial framework based off of the PyTorch DQN intermediate tutorial (see here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings

from envs.test_env import TestEnv
from utils.DQN import DQN
from utils.replay_memory import ReplayMemory
from utils.transition import Transition

warnings.filterwarnings("ignore", category=DeprecationWarning, message="`np.bool8` is a deprecated alias for `np.bool_`")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    def __init__(self, env,
                    batch_size:int  = 128,
                    gamma:float     = 0.99,
                    eps_start:float = 0.9,
                    eps_end:float   = 0.05,
                    eps_decay:int   = 1000,
                    tau:float       = 0.005,
                    lr:float        = 1e-4
                ) -> None:
        
        self.env        = env
        self.batch_size = batch_size
        self.gamma      = gamma
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.tau        = tau
        self.lr         = lr
        
        state, _ = self.env.reset()
        n_observations = len(state)
        n_actions = self.env.action_space.n
        
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        
    def select_action(self, state:torch.Tensor) -> torch.Tensor:
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)
    
    def optimize(self) -> None:
        if len(self.memory) < self.batch_size:
            return
            
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def train(self, num_episodes:int=600) -> None:
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            steps = 0
            
            while True:
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated
                
                next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                self.memory.push(state, action, next_state, reward)
                state = next_state
                
                self.optimize()
                
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)
                
                steps += 1
                if done:
                    print(f"Episode {episode} - Steps: {steps}")
                    break
    