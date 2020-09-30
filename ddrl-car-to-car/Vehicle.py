import random


class Vehicle:
    MAX_STEER = 1
    MIN_STEER = -1

    def __init__(self, model="vehicle.tesla.model3"):
        self.vehicle_model = model

    def get_random_action(self):
        throttle = random.random()
        accelerate = random.random() > 0.4
        return {
            "throttle": throttle if accelerate else 0,
            "steer": random.uniform(self.MIN_STEER, self.MAX_STEER),
            "brake": 0 if accelerate else throttle,
            "handbrake": random.random() > 0.8,
            "reverse": random.random() > 0.6
        }
