
from agents.default import DefaultAgent
import traci
import wandb

# This agent changes phase after a fixed duration:
class FixedDurationAgent(DefaultAgent):

    # The agent is initialized with the environment:
    def __init__(self, env, duration, wandb_on=False) -> None:
        super().__init__(env, wandb_on)
        self.phase = 0
        self.duration = duration
        
    # The agent selects and action (the next phase in the sequence):
    def select_action(self) -> int:
        self.phase = (self.phase + 1) % len(self.env.phases)
        return self.phase

    # The agent runs the environment for a specified number of episodes:
    def run(self, num_episodes:int=1, sumo_gui:bool=True) -> None:
        for episode in range(num_episodes):
            self.env.reset(sumo_gui)
            while True:
                if traci.simulation.getTime() % self.duration == 0:
                    action = self.select_action()
                else:
                    action = None
                _, _, terminated, env_info = self.env.step(action)

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
        