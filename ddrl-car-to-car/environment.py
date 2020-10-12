import carla
import time
import math
from carla_utils.world import World
from carla_utils.actors import RGBCamera, CollisionDetector
from carla_utils import config
import random


class CarlaEnvironment:
    STEER_AMT = 1.0

    def __init__(self,
                 camera_config=(
                         config.DEFAULT_RGB_CAMERA_IM_WIDTH,
                         config.DEFAULT_RGB_CAMERA_IM_HEIGHT,
                         config.DEFAULT_RGB_CAMERA_FOV)):
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

    def step(self, action):
        if action == 0:
            self.vehicle.move(throttle=1.0, steer=-1 * self.STEER_AMT)
        elif action == 1:
            self.vehicle.move(throttle=1.0, steer=0)
        elif action == 2:
            self.vehicle.move(throttle=1.0, steer=1 * self.STEER_AMT)

        v = self.vehicle.vehicle_actor.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        if len(self.collision_detector.data) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        return self.get_current_state(), reward, done, None

    def destroy(self):
        for a in self.actor_list:
            a.destroy()
        self.actor_list = []

    def move_view_to_vehicle_position(self):
        spectator = self.carla_world.world.get_spectator()
        transform = self.vehicle.vehicle_actor.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))