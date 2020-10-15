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
        self.car_is_stopped_since = 0

    def reset(self):
        self.actor_list.clear()

        vehicle_location = random.choice(self.carla_world.world.get_map().get_spawn_points())
        self.vehicle = self.carla_world.create_vehicle(position=vehicle_location)
        self.actor_list.append(self.vehicle.vehicle_actor)

        self.vehicle.stay_still()
        self.attach_sensor_to_vehicle(self.front_camera)

        self.collision_detector.data.clear()
        self.attach_sensor_to_vehicle(self.collision_detector)

        self.lane_invasion_detector.data.clear()
        self.attach_sensor_to_vehicle(self.lane_invasion_detector)

        self.wait_environment_ready()

    def wait_environment_ready(self):
        while any(x is None for x in self.actor_list):
            time.sleep(0.01)

    def attach_sensor_to_vehicle(self, sensor):
        self.carla_world.attach_sensor_to_vehicle(self.vehicle, sensor)
        self.actor_list.append(sensor.sensor_actor)

    def get_current_state(self):
        return self.front_camera.data

    def step(self, choice):
        print(choice)
        action = self.last_action
        if choice < len(self.actions):
            action = self.actions[choice]
            self.vehicle.move(throttle=action[0], steer=action[1], brake=action[2])

        v = self.vehicle.vehicle_actor.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        speed_limit = self.vehicle.vehicle_actor.get_speed_limit()
        min_speed_limit = speed_limit * MINIMUM_SPEED_LIMIT_PERCENTAGE
        if self.car_is_stopped_since == -1 and kmh <= 0:
            self.car_is_stopped_since = time.time()
        elif kmh > 0:
            self.car_is_stopped_since = -1

        reward = 0
        if len(self.collision_detector.data) != 0:
            done = True
            reward = CRASH
        else:
            done = False
            is_at_traffic_light_red = self.vehicle.vehicle_actor.get_traffic_light_state() == TrafficLightState.Red
            if kmh > speed_limit:
                reward += OVER_SPEED_LIMIT
            elif kmh < min_speed_limit and not is_at_traffic_light_red:
                reward += UNDER_MINIMUM_SPEED_LIMIT
            elif kmh >= min_speed_limit:
                reward += IN_SPEED_LIMIT
            if self.car_is_stopped_since != -1 and not is_at_traffic_light_red:
                stopped_time = time.time() - self.car_is_stopped_since
                if stopped_time >= MAX_STOPPED_SECONDS_ALLOWED:
                    reward += STOPPED_TOO_LONG
            reward += TURN if action[1] != 0 else FORWARD
            if self.last_action[1] * action[1] < 0:
                reward += OPPOSITE_TURN
            if is_at_traffic_light_red and (action[0] > 0 or (action[2] <= 0 and kmh > 0)):
                reward += FORWARD_AT_INTERSECTION_RED
            elif is_at_traffic_light_red and action[0] == 0 and (action[2] > 0 or kmh <= 0):
                reward += STOP_AT_INTERSECTION_RED
            if len(self.lane_invasion_detector.data) > 0:
                lane_changes_not_allowed = any(self.is_lane_change_not_allowed(x.lane_change)
                                               for x in self.lane_invasion_detector.data)
                if lane_changes_not_allowed:
                    waypoint = self.carla_world.get_map().get_waypoint(
                        self.vehicle.get_location(),
                        project_to_road=True,
                        lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk)
                    )
                    # se driving => contromano | corsia sbagliata
                    # `se shoulder | sidewalk => siamo fuori strada
                    if waypoint.lane_type == carla.LaneType.Driving:
                        reward += WRONG_SIDE_ROAD
                    elif waypoint.lane_type == carla.LaneType.Shoulder or waypoint.lane_type == carla.LaneType.Sidewalk:
                        reward += OFF_ROAD
                    else:
                        reward += WRONG_SIDE_ROAD
                else:
                    # superato linea permessa centrale ma siamo contro mano
                    pass

        self.last_action = action
        return self.get_current_state(), reward, done, None

    def is_lane_change_not_allowed(self, lane_change):
        steer = self.last_action[1]
        return lane_change == carla.LaneChange.NONE \
               or (lane_change == carla.LaneChange.Right and steer < 0) \
               or (lane_change == carla.LaneChange.Left and steer > 0)

    def destroy(self):
        for a in self.actor_list:
            a.destroy()
        self.actor_list.clear()

    def move_view_to_vehicle_position(self):
        spectator = self.carla_world.world.get_spectator()
        transform = self.vehicle.vehicle_actor.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
