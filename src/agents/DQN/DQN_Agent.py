
# This iteration of DQN is a simple initial framework based off of the PyTorch DQN intermediate tutorial (see here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import warnings
from pathlib import Path

from utils.NN import NN
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
                    eps_decay:int   = 10000,
                    tau:float       = 0.005,
                    lr:float        = 1e-4,
                    wandb_on:bool      = False
                ) -> None:
        
        self.env        = env
        self.batch_size = batch_size
        self.gamma      = gamma
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.eps        = 0
        self.tau        = tau
        self.lr         = lr
        self.wandb_on   = wandb_on

        if wandb_on:
            wandb.init(project="DQN-Training", config={
                "batch_size" : batch_size,
                "gamma"      : gamma,
                "eps_start"  : eps_start,
                "eps_end"    : eps_end,
                "eps_decay"  : eps_decay,
                "tau"        : tau,
                "lr"         : lr
            })
        
        n_observations = len(self.env.observation_space)
        n_actions = len(self.env.action_space)
        
        self.policy_net = NN(n_observations, n_actions).to(device)
        self.target_net = NN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(100000)
        self.steps_done = 0
        
    def select_action(self, state:torch.Tensor) -> torch.Tensor:
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > self.eps:
            with torch.no_grad():
                output = self.policy_net(state).argmax(1).unsqueeze(0)
                return output
        return torch.tensor([[random.choice((0, 1))]], device=device, dtype=torch.long)
    
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
        
    def train(self, num_episodes:int=100, sumo_gui=False) -> None:
        rewards = []
        for episode in range(num_episodes + 1):
            log = (episode % 100 == 0)
            state = self.env.reset(sumo_gui)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            steps = 0
            total_reward = 0
            
            while True:
                action = self.select_action(state)
                observation, reward, terminated, total_waiting_time, average_queue_length_N, average_queue_length_S = self.env.step(action)
                reward = torch.tensor([reward], device=device)
                total_reward += reward.item()
                
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
                if terminated:
                    rewards.append(total_reward)
                    # print(f"Episode {episode} - Steps: {steps} - Epsilon: {self.eps}")
                    print(f"Episode: {episode} - Steps: {steps} - TWT: {total_waiting_time}")
                    if self.wandb_on:
                        wandb.log({
                            "episode": episode, 
                            "reward": total_reward, 
                            "steps": steps, 
                            "TWT": total_waiting_time, 
                            "rolling average": sum(rewards[-100:])/100 if len(rewards) >= 100 else None,
                            "AQL North": average_queue_length_N,
                            "AQL South": average_queue_length_S
                        })

                    if log:
                        self.save(f"DQN_{episode}.pth")
                    break
                
        self.env.close()
        if self.wandb_on:
            wandb.finish()

    def save(self, filename:str, path="saved/models") -> None:
        save_path = Path(path) / filename
        torch.save(self.policy_net.state_dict(), save_path)

    def load(self, filename:str, path="saved/models") -> None:
        self.policy_net.load_state_dict(torch.load(Path(path) / filename, weights_only=True))

    def run(self, num_episodes:int=10, sumo_gui=False) -> None:
        for episode in range(num_episodes):
            state = self.env.reset(sumo_gui)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            steps = 0
            
            while True:
                action = self.policy_net(state).max(1)[1].view(1, 1)
                observation, _, terminated, total_waiting_time, _, _ = self.env.step(action.item())
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                
                steps += 1
                if terminated:
                    print(f"Episode {episode} - Steps: {steps} - TWT: {total_waiting_time}")
                    break
        
        self.env.close()
        