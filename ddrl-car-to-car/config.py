from carla_utils.actors import RGBCamera

RGB_CAMERA_FOV = 110
RGB_CAMERA_IM_WIDTH = 640
RGB_CAMERA_IM_HEIGHT = 480

IMG_DIMENSION = (RGB_CAMERA_IM_HEIGHT, RGB_CAMERA_IM_WIDTH, RGBCamera.get_number_of_channels())

FPS = 60

THROTTLE = [0, 0.5, 1]
STEER = [-1, -0.5, 0, 0.5, 1]
BRAKE = [0, 0.5, 1]
