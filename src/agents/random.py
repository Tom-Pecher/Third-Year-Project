
from agents.default import DefaultAgent
import random
import traci
import wandb

# This agent changes phase randomly at each step:
class RandomAgent(DefaultAgent):

    # The agent is initialized with the environment:
    def __init__(self, env, probablity, wandb_on=False) -> None:
        super().__init__(env, wandb_on)
        self.probability = probablity
        
    # The agent selects and action (by random):
    def select_action(self) -> int:
        self.phase = random.randrange(0, len(self.env.phases))
        return self.phase

    # The agent runs the environment for a specified number of episodes:
    def run(self, num_episodes:int=1, sumo_gui=True) -> None:
        for episode in range(num_episodes):
            self.env.reset(sumo_gui)
            while True:
                if random.random() < self.probability:
                    action = self.select_action()
                else:
                    action = None
                _, _, terminated, env_info = self.env.step(action)

                # Log metrics:
                if self.wandb_on:
                    wandb.log({
                        "episode": episode,
                        "step": traci.simulation.getTime(),
                        **env_info
                    })

                if terminated:
                    print(f"Episode {episode} - COMPLETE")
                    break
        
        self.env.close()
        