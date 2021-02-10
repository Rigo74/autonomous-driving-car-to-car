from collections import deque

import numpy as np
import carla
import cv2
from carla_utils.config import *


class Vehicle:
    MAX_STEER = 1
    MIN_STEER = -1

    def __init__(self, vehicle_actor):
        self.vehicle_actor = vehicle_actor

    def stay_still(self):
        self.move()

    def move(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        self.vehicle_actor.apply_control(
            carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=hand_brake,
                reverse=reverse
            )
        )

    def get_speed(self):
        return self.vehicle_actor.get_velocity()

    def get_speed_limit(self):
        return self.vehicle_actor.get_speed_limit()

    def get_location(self):
        return self.vehicle_actor.get_location()


class Sensor:

    def __init__(self, blueprint_library, model, location=carla.Location(), attributes=[]):
        self.blueprint = blueprint_library.find(model)
        for attr in attributes:
            self.blueprint.set_attribute(attr[0], attr[1])
        self.location = carla.Transform(location)
        self.sensor_actor = None
        self.data = None

    def callback(self, data):
        raise NotImplemented()


class RGBCamera(Sensor):
    SHOW_CAM = False
    MODEL = "sensor.camera.rgb"

    def __init__(self, blueprint_library, location, attributes=[],
                 im_width=DEFAULT_RGB_CAMERA_IM_WIDTH,
                 im_height=DEFAULT_RGB_CAMERA_IM_HEIGHT):
        super().__init__(blueprint_library, self.MODEL, location, attributes)
        self.im_height = im_height
        self.im_width = im_width
        self.data = []
        self.channels = RGBCamera.get_number_of_channels()

    @staticmethod
    def create(blueprint_library, location,
               im_width=DEFAULT_RGB_CAMERA_IM_WIDTH,
               im_height=DEFAULT_RGB_CAMERA_IM_HEIGHT,
               fov=DEFAULT_RGB_CAMERA_FOV):
        attributes = [
            ("image_size_x", f"{im_width}"),
            ("image_size_y", f"{im_height}"),
            ("fov", f"{fov}")
        ]
        return RGBCamera(
            blueprint_library=blueprint_library,
            location=location,
            attributes=attributes,
            im_width=im_width,
            im_height=im_height
        )

    @staticmethod
    def create_default(blueprint_library, location):
        return RGBCamera.create(blueprint_library, location)

    @staticmethod
    def get_number_of_channels():
        return 3

    # Override
    def callback(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.data = i3


class RGBCameraMultiplePhoto(RGBCamera):

    def __init__(self, blueprint_library, location, attributes=[],
                 im_width=DEFAULT_RGB_CAMERA_IM_WIDTH,
                 im_height=DEFAULT_RGB_CAMERA_IM_HEIGHT,
                 im_history_len=DEFAULT_RGB_CAMERA_HISTORY_LENGTH):
        super().__init__(blueprint_library, location, attributes, im_width, im_height)
        self.data = deque(maxlen=im_history_len)

    @staticmethod
    def create(blueprint_library, location,
               im_width=DEFAULT_RGB_CAMERA_IM_WIDTH,
               im_height=DEFAULT_RGB_CAMERA_IM_HEIGHT,
               fov=DEFAULT_RGB_CAMERA_FOV,
               im_history_len=DEFAULT_RGB_CAMERA_HISTORY_LENGTH):
        attributes = [
            ("image_size_x", f"{im_width}"),
            ("image_size_y", f"{im_height}"),
            ("fov", f"{fov}")
        ]
        return RGBCameraMultiplePhoto(
            blueprint_library=blueprint_library,
            location=location,
            attributes=attributes,
            im_width=im_width,
            im_height=im_height,
            im_history_len=im_history_len
        )

    @staticmethod
    def create_default(blueprint_library, location):
        return RGBCameraMultiplePhoto.create(blueprint_library, location)

    # Override
    def callback(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        if self.SHOW_CAM:
            cv2.imshow("", i2[:, :, :3])
            cv2.waitKey(1)
        self.data.append([i2[:, :, i] for i in range(0, 3)])  # 4, 3, h, w


class GreyScaleCamera(RGBCamera):

    def __init__(self, blueprint_library, location, attributes=[],
                 im_width=DEFAULT_RGB_CAMERA_IM_WIDTH,
                 im_height=DEFAULT_RGB_CAMERA_IM_HEIGHT):
        super().__init__(blueprint_library, location, attributes, im_width, im_height)
        self.channels = GreyScaleCamera.get_number_of_channels()

    @staticmethod
    def create(blueprint_library, location,
               im_width=DEFAULT_RGB_CAMERA_IM_WIDTH,
               im_height=DEFAULT_RGB_CAMERA_IM_HEIGHT,
               fov=DEFAULT_RGB_CAMERA_FOV):
        attributes = [
            ("image_size_x", f"{im_width}"),
            ("image_size_y", f"{im_height}"),
            ("fov", f"{fov}")
        ]
        return GreyScaleCamera(
            blueprint_library=blueprint_library,
            location=location,
            attributes=attributes,
            im_width=im_width,
            im_height=im_height
        )

    @staticmethod
    def create_default(blueprint_library, location):
        return GreyScaleCamera.create(blueprint_library, location)

    @staticmethod
    def get_number_of_channels():
        return 1

    # Override
    def callback(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = cv2.cvtColor(i2[:, :, :3], cv2.COLOR_BGR2GRAY)
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.data = np.array(i3)


class CollisionDetector(Sensor):
    MODEL = "sensor.other.collision"

    def __init__(self, blueprint_library):
        super().__init__(blueprint_library, self.MODEL)
        self.data = []

    # Override
    def callback(self, event):
        self.data.append(event)


class LaneInvasionDetector(Sensor):
    MODEL = "sensor.other.lane_invasion"

    def __init__(self, blueprint_library):
        super().__init__(blueprint_library, self.MODEL)
        self.data = []

    # Override
    def callback(self, lane_invasion_event):
        self.data += lane_invasion_event.crossed_lane_markings
