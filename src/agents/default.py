
# This is a default agent from which all other agent's inherit from.

import traci
import wandb

# The default agent selects the first action in the action space at each time step:
class DefaultAgent:

    # The agent is initialized with the environment:
    def __init__(self, env, wandb_on=False) -> None:
        self.env = env
        self.wandb_on = wandb_on
        self.phase = 0

        # Initialize wandb
        if wandb_on:
            wandb.init(
                project="traffic-rl",
                config={
                    "agent": self.__class__.__name__,
                    "env": self.env.__class__.__name__
                }
            )
        
    # The agent selects and action (by default, the first action in the action space):
    def select_action(self) -> int:
        return self.phase

    # The agent runs the environment for a specified number of episodes:
    def run(self, num_episodes:int=1, sumo_gui=True) -> None:
        for episode in range(num_episodes):
            self.env.reset(sumo_gui)
            while True:
                action = self.select_action()
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
        