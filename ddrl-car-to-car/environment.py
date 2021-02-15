import carla
import time
import math
import random

from carla_utils.world import World
from carla_utils.actors import CollisionDetector, LaneInvasionDetector, RGBCameraMultiplePhoto
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


TURN_THRESHOLD = 0.75


class CarlaEnvironment:
    STEER_AMT = 1.0

    def __init__(self,
                 camera_config=(RGB_CAMERA_IM_WIDTH, RGB_CAMERA_IM_HEIGHT, RGB_CAMERA_FOV, RGB_CAMERA_HISTORY)
                 ):
        self.carla_world = World()
        self.actor_list = []
        self.vehicle = None
        self.front_camera = GreyScaleCamera.create(
            self.carla_world.blueprint_library,
            carla.Location(x=2.2, z=0.7),
            im_width=camera_config[0],
            im_height=camera_config[1],
            fov=camera_config[2]
        )
        self.collision_detector = CollisionDetector(self.carla_world.blueprint_library)
        self.lane_invasion_detector = LaneInvasionDetector(self.carla_world.blueprint_library)
        self.actions = create_actions_list()
        self.last_action = (0, 0, 0)
        self.car_is_stopped_since = -1
        self.car_is_steering_since = -1
        self.carla_world.load_map("Town03")

    def reset(self, change_map=False):
        self.destroy()

        if change_map:
            self.carla_world.load_map()

        sps = self.carla_world.get_spawn_points()
        vehicle_location = random.choice(sps) if random.random() > TURN_THRESHOLD \
            else sps[random.choice(self.carla_world.get_turns_spawn_points_indexes())]

        self.vehicle = self.carla_world.create_vehicle(position=vehicle_location)
        self.actor_list.append(self.vehicle.vehicle_actor)

        self.vehicle.stay_still()
        # self.front_camera.data.clear()
        self.attach_sensor_to_vehicle(self.front_camera)

        self.collision_detector.data.clear()
        self.attach_sensor_to_vehicle(self.collision_detector)

        self.lane_invasion_detector.data.clear()
        self.attach_sensor_to_vehicle(self.lane_invasion_detector)

        self.wait_environment_ready()

        self.car_is_stopped_since = -1
        self.car_is_steering_since = -1

    def wait_environment_ready(self):
        while any(x is None for x in self.actor_list):
            time.sleep(0.1)
        # while len(self.front_camera.data) < self.front_camera.data.maxlen:
            # time.sleep(0.1)

    def attach_sensor_to_vehicle(self, sensor):
        self.carla_world.attach_sensor_to_vehicle(self.vehicle, sensor)
        self.actor_list.append(sensor.sensor_actor)

    def get_current_state(self):
        return self.front_camera.data

    def get_number_of_actions(self):
        return len(self.actions)

    def do_action(self, choice):
        action = self.last_action
        if choice < len(self.actions):
            action = self.actions[choice]
            self.vehicle.move(throttle=action[0], steer=action[1], brake=action[2])
        return action

    def step(self, choice):
        action = self.do_action(choice)
        lane_invasion_data = self.lane_invasion_detector.data.copy()
        self.lane_invasion_detector.data.clear()

        current_speed = self.evaluate_speed_values()

        reward = 0
        done = False
        if len(self.collision_detector.data) != 0:
            done = True
            reward = CRASH
            # print(f"[CRASH] {CRASH}")
        else:
            speed_reward = self.evaluate_speed_reward()
            # print(f"[SPEED_REWARD] {speed_reward}")
            reward += speed_reward

            action_reward = self.evaluate_action_reward(
                current_action=action,
                current_speed=current_speed
            )
            # print(f"[ACTION_REWARD] {action_reward}")
            reward += action_reward

            if len(lane_invasion_data) > 0:
                done, crossing_line_reward = self.evaluate_crossing_line_reward()
                # print(f"[CROSSING_LINE_REWARD] {crossing_line_reward}")
                reward = crossing_line_reward if done else (reward + crossing_line_reward)

        # print("---------------------------------------------------------------")
        self.last_action = action
        return self.get_current_state(), reward, done

    def evaluate_speed_values(self):
        v = self.vehicle.get_speed()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        if self.car_is_stopped_since == -1 and kmh <= 0:
            self.car_is_stopped_since = time.time()
        elif kmh > 0:
            self.car_is_stopped_since = -1
        return kmh

    def evaluate_speed_reward(self):
        if self.car_is_stopped_since != -1:
            stopped_time = time.time() - self.car_is_stopped_since
            if stopped_time >= MAX_STOPPED_SECONDS_ALLOWED:
                return STOPPED_TOO_LONG
        return 0

    def evaluate_action_reward(self, current_action, current_speed):
        steer = current_action[1]
        is_opposite_turn = self.last_action[1] * steer < 0

        if steer != 0:
            if self.last_action[1] * steer > 0 and self.car_is_steering_since != -1:
                steering_time = time.time() - self.car_is_steering_since
                if steering_time >= MAX_STEERING_SECONDS_ALLOWED:
                    return STEERING_TOO_LONG
            else:
                self.car_is_steering_since = time.time()
        else:
            self.car_is_steering_since = -1

        if current_speed > 0 and not is_opposite_turn:
            return TURN if steer != 0 else FORWARD
        else:
            return OPPOSITE_TURN if is_opposite_turn else 0

    def evaluate_crossing_line_reward(self):
        waypoint = self.carla_world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=None,
            lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk | carla.LaneType.Shoulder)
        )
        if waypoint is not None \
                and (waypoint.lane_type == carla.LaneType.Shoulder or waypoint.lane_type == carla.LaneType.Sidewalk):
            return True, OFF_ROAD
        return False, 0

    def destroy(self):
        for a in self.actor_list:
            a.destroy()
        self.actor_list.clear()

    def move_view_to_vehicle_position(self):
        spectator = self.carla_world.world.get_spectator()
        transform = self.vehicle.vehicle_actor.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
