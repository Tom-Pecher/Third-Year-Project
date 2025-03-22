
import traci

# The vehicle class stores valuable information about the vehicles in the simulation.
class Vehicle:
    def __init__(self, id:str) -> None:
        self.id = id
        self.in_simulation = True
        self.pos = 0
        self.speed = 0
        self.waiting_time = 0
        self.emergency_brakes = 0
        self.time_loss = 0
        self.cumulative_time_loss = 0

        # Tracking values:
        self.last_waiting_time = 0
        self.last_in_simulation = True
        self.cumulative_waiting_time = 0
        self.last_cumulative_time_loss = 0
        
    # Convert the vehicle to a string (for easy printing):
    def __str__(self) -> str:
        return self.id
    
    # Update the vehicle's information:
    def update(self) -> None:
        self.pos = traci.vehicle.getPosition(self.id)
        self.speed = traci.vehicle.getSpeed(self.id)
        self.waiting_time = traci.vehicle.getWaitingTime(self.id)

        self.time_loss = traci.vehicle.getTimeLoss(self.id) - self.last_cumulative_time_loss
        self.last_cumulative_time_loss = traci.vehicle.getTimeLoss(self.id)

        if traci.vehicle.getAcceleration(self.id) <= -8:
            self.emergency_brakes += 1

        # Update cumulative values:
        if self.last_waiting_time != 0 and self.waiting_time == 0:
            self.cumulative_waiting_time += self.last_waiting_time
            self.last_waiting_time = 0
        else:
            self.last_waiting_time = self.waiting_time
        
        self.cumulative_time_loss = traci.vehicle.getTimeLoss(self.id)
