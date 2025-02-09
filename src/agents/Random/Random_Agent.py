import random

class RandomAgent:
    def __init__(self, env,
                ) -> None:
        
        self.env        = env
        self.steps_done = 0
        
    def select_action(self) -> int:
        return random.randint(0, 1)

    def run(self, num_episodes:int=10, sumo_gui=False) -> None:
        for episode in range(num_episodes):
            self.steps_done = 0
            action = 0
            self.env.reset(sumo_gui)
            while True:
                if self.steps_done % 10 == 0:
                    action = self.select_action()
                _, _, terminated, total_waiting_time = self.env.step(action)
                
                self.steps_done += 1
                if terminated:
                    print(f"Episode {episode} - Steps: {self.steps_done} - TWT: {total_waiting_time}")
                    break
        
        self.env.close()
        