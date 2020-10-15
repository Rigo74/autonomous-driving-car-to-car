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


def create_actions_list():
    actions = set()
    for s in STEER:
        for t in THROTTLE:
            actions.add((t, s, 0))
        for b in BRAKE:
            actions.add((0, s, b))
    return list(actions)


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
        self.actions = create_actions_list()
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

    def get_number_of_actions(self):
        return len(self.actions) + 1

    def step(self, choice):
        action = self.last_action
        if choice < len(self.actions):
            action = self.actions[choice]
            self.vehicle.move(throttle=action[0], steer=action[1], brake=action[2])

        current_speed, speed_limit, min_speed_limit = self.evaluate_speed_values()

        reward = 0
        if len(self.collision_detector.data) != 0:
            done = True
            reward = CRASH
        else:
            done = False
            is_at_traffic_light_red = self.vehicle.vehicle_actor.get_traffic_light_state() == TrafficLightState.Red

            reward += self.evaluate_speed_reward(
                current_speed=current_speed,
                speed_limit=speed_limit,
                min_speed_limit=min_speed_limit,
                is_at_traffic_light_red=is_at_traffic_light_red
            )

            reward += self.evaluate_action_reward(
                current_action=action,
                current_speed=current_speed,
                is_at_traffic_light_red=is_at_traffic_light_red
            )

            if len(self.lane_invasion_detector.data) > 0:
                reward += self.evaluate_reward_for_crossing_line()

        self.last_action = action
        return self.get_current_state(), reward, done

    def evaluate_speed_values(self):
        v = self.vehicle.get_speed()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        speed_limit = self.vehicle.get_speed_limit()
        min_speed_limit = speed_limit * MINIMUM_SPEED_LIMIT_PERCENTAGE
        if self.car_is_stopped_since == -1 and kmh <= 0:
            self.car_is_stopped_since = time.time()
        elif kmh > 0:
            self.car_is_stopped_since = -1
        return kmh, speed_limit, min_speed_limit

    def evaluate_speed_reward(self, current_speed, speed_limit, min_speed_limit, is_at_traffic_light_red):
        if current_speed > speed_limit:
            return OVER_SPEED_LIMIT
        elif current_speed >= min_speed_limit:
            return IN_SPEED_LIMIT

        reward = UNDER_MINIMUM_SPEED_LIMIT \
            if (current_speed < min_speed_limit and not is_at_traffic_light_red) \
            else 0
        if self.car_is_stopped_since != -1 and not is_at_traffic_light_red:
            stopped_time = time.time() - self.car_is_stopped_since
            if stopped_time >= MAX_STOPPED_SECONDS_ALLOWED:
                reward += STOPPED_TOO_LONG
        return reward

    '''
    if current_speed > speed_limit:
        reward += OVER_SPEED_LIMIT
    elif current_speed < min_speed_limit and not is_at_traffic_light_red:
        reward += UNDER_MINIMUM_SPEED_LIMIT
    elif current_speed >= min_speed_limit:
        reward += IN_SPEED_LIMIT
    if self.car_is_stopped_since != -1 and not is_at_traffic_light_red:
        stopped_time = time.time() - self.car_is_stopped_since
        if stopped_time >= MAX_STOPPED_SECONDS_ALLOWED:
            reward += STOPPED_TOO_LONG
    '''

    def evaluate_action_reward(self, current_action, current_speed, is_at_traffic_light_red):
        throttle = current_action[0]
        steer = current_action[1]
        brake = current_action[2]
        partial_reward = OPPOSITE_TURN if (self.last_action[1] * steer < 0) else 0
        if is_at_traffic_light_red and (throttle > 0 or (brake <= 0 and current_speed > 0)):
            return partial_reward + FORWARD_AT_INTERSECTION_RED
        elif is_at_traffic_light_red and throttle == 0 and (brake > 0 or current_speed <= 0):
            return partial_reward + STOP_AT_INTERSECTION_RED
        elif steer != 0:
            return TURN
        else:
            return FORWARD

    '''
        if self.last_action[1] * action[1] < 0:
            reward += OPPOSITE_TURN
        elif action[1] != 0:
            reward += TURN
        else:
            reward += FORWARD
        if is_at_traffic_light_red and (action[0] > 0 or (action[2] <= 0 and current_speed > 0)):
            reward += FORWARD_AT_INTERSECTION_RED
        elif is_at_traffic_light_red and action[0] == 0 and (action[2] > 0 or current_speed <= 0):
            reward += STOP_AT_INTERSECTION_RED
    '''

    def evaluate_reward_for_crossing_line(self):
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
                return WRONG_SIDE_ROAD
            elif waypoint.lane_type == carla.LaneType.Shoulder or waypoint.lane_type == carla.LaneType.Sidewalk:
                return OFF_ROAD
            else:
                return WRONG_SIDE_ROAD
        else:
            # superato linea permessa centrale ma siamo contro mano
            return 0

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
