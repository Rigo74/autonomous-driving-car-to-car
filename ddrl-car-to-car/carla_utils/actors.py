import numpy as np
import carla
import cv2
from carla_utils import config


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


class Sensor:

    def __init__(self, blueprint_library, model, location=carla.Location(), attributes=[]):
        self.blueprint = blueprint_library.find(model)
        for attr in attributes:
            self.blueprint.set_attribute(attr[0],attr[1])
        self.location = carla.Transform(location)
        self.sensor_actor = None

    def callback(self, data):
        raise NotImplemented()


class RGBCamera(Sensor):
    SHOW_CAM = False
    MODEL = "sensor.camera.rgb"

    def __init__(self, blueprint_library, location, attributes=[],
                 im_width=config.DEFAULT_RGB_CAMERA_IM_WIDTH, im_height=config.DEFAULT_RGB_CAMERA_IM_HEIGHT):
        super().__init__(blueprint_library, self.MODEL, location, attributes)
        self.im_height = im_height
        self.im_width = im_width
        self.data = []

    @staticmethod
    def create_default(blueprint_library, location):
        attributes = [
            ("image_size_x", f"{config.DEFAULT_RGB_CAMERA_IM_WIDTH}"),
            ("image_size_y", f"{config.DEFAULT_RGB_CAMERA_IM_HEIGHT}"),
            ("fov", f"{config.DEFAULT_RGB_CAMERA_FOV}")
        ]
        return RGBCamera(blueprint_library, location, attributes)

    # Override
    def callback(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.data = i3


class CollisionDetector(Sensor):
    MODEL = "sensor.other.collision"

    def __init__(self, blueprint_library):
        super().__init__(blueprint_library, self.MODEL)
        self.data = []

    # Override
    def callback(self, event):
        self.data.append(event)
