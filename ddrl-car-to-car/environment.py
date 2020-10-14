import carla
import time
import math
import random
from carla import TrafficLightState

from carla_utils.world import World
from carla_utils.actors import CollisionDetector, LaneInvasionDetector
from carla_utils.config import *
from config import *
from rewards import *


class CarlaEnvironment:
    STEER_AMT = 1.0

    def __init__(self,
                 camera_config=(DEFAULT_RGB_CAMERA_IM_WIDTH, DEFAULT_RGB_CAMERA_IM_HEIGHT, DEFAULT_RGB_CAMERA_FOV)):
        self.carla_world = World()
        self.episode_start = time.time()
        self.actor_list = []
        self.vehicle = None
        self.front_camera = RGBCamera.create(
            self.carla_world.blueprint_library,
            carla.Location(x=2.2, z=0.7),
            im_width=camera_config[0],
            im_height=camera_config[1],
            fov=camera_config[2]
        )
        self.collision_detector = CollisionDetector(self.carla_world.blueprint_library)
        self.lane_invasion_detector = LaneInvasionDetector(self.carla_world.blueprint_library)
        actions = set()
        for s in STEER:
            for t in THROTTLE:
                actions.add((t, s, 0))
            for b in BRAKE:
                actions.add((0, s, b))
        self.actions = list(actions)
        self.last_action = (0, 0, 0)

    def reset(self):
        self.actor_list = []

        vehicle_location = random.choice(self.carla_world.world.get_map().get_spawn_points())
        self.vehicle = self.carla_world.create_vehicle(position=vehicle_location)
        self.actor_list.append(self.vehicle.vehicle_actor)

        self.vehicle.stay_still()
        time.sleep(4)

        self.carla_world.attach_sensor_to_vehicle(self.vehicle, self.front_camera)
        self.actor_list.append(self.front_camera.sensor_actor)

        self.collision_detector.data = []
        self.carla_world.attach_sensor_to_vehicle(self.vehicle,self.collision_detector)
        self.actor_list.append(self.collision_detector.sensor_actor)

        while self.front_camera.sensor_actor is None:
            time.sleep(0.01)

        self.vehicle.stay_still()

    def get_current_state(self):
        return self.front_camera.data

    def step(self, choice):
        action = self.last_action
        if choice < len(self.actions):
            action = self.actions[choice]
            self.vehicle.move(throttle=action[0], steer=action[1], brake=action[2])

        v = self.vehicle.vehicle_actor.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        speed_limit = self.vehicle.vehicle_actor.get_speed_limit()

        reward = 0
        if len(self.collision_detector.data) != 0:
            done = True
            reward = CRASH
        else:
            done = False
            reward += IN_SPEED_LIMIT if kmh <= speed_limit else OVER_SPEED_LIMIT
            reward += TURN if action[1] != 0 else FORWARD
            if self.last_action[1] * action[1] < 0:
                reward += OPPOSITE_TURN
            is_at_traffic_light_red = self.vehicle.vehicle_actor.get_traffic_light_state() == TrafficLightState.Red
            if is_at_traffic_light_red and (action[0] > 0 or (action[2] <= 0 and kmh > 0)):
                reward += FORWARD_AT_INTERSECTION_RED
            elif is_at_traffic_light_red and action[0] == 0 and (action[2] > 0 or kmh <= 0):
                reward += STOP_AT_INTERSECTION_RED


        self.last_action = action
        return self.get_current_state(), reward, done, None

    def destroy(self):
        for a in self.actor_list:
            a.destroy()
        self.actor_list = []

    def move_view_to_vehicle_position(self):
        spectator = self.carla_world.world.get_spectator()
        transform = self.vehicle.vehicle_actor.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
