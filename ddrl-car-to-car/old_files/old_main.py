import carla
import time
from carla.Vehicle import Vehicle
import numpy as np
import cv2

CLIENT_ADDRESS = "localhost"
CLIENT_PORT = 2000

IM_WIDTH = 640
IM_HEIGHT = 480

actors = []


def create_client(host="localhost", port=2000):
    _client = carla.Client(host, port)
    _client.set_timeout(10.0)
    _world = _client.get_world()
    _world = _client.reload_world()
    # _world.set_weather(carla_utils.WeatherParameters.ClearNoon)
    '''
    settings = _world.get_settings()
    settings.fixed_delta_seconds = 1.0/2
    _world.apply_settings(settings)
    '''
    return _client, _world


def move_view_to_vehicle_position(world, vehicle):
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))


def cameraData(image, l_images):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # RGBA
    i3 = i2[:, :, :3]
    l_images.append(i3)
    return


try:
    client, world = create_client(host=CLIENT_ADDRESS, port=CLIENT_PORT)
    vehicle = Vehicle("tt")

    blueprint_library = world.get_blueprint_library()
    vehicle_blueprint = blueprint_library.filter(vehicle.vehicle_model)[0]

    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle.vehicle_actor = world.spawn_actor(vehicle_blueprint, spawn_point)
    move_view_to_vehicle_position(world, vehicle.vehicle_actor)

    l_images = []

    # frontal camera
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")

    # x=profondità y=destra_sinistra z=altezza
    cam_location = carla.Transform(carla.Location(x=2.2, z=0.7))
    cam_sensor = world.spawn_actor(cam_bp, cam_location, attach_to=vehicle.vehicle_actor)

    cam_sensor.listen(lambda data: cameraData(data, l_images))

    # lane invasion
    lane_bp = blueprint_library.find("sensor.other.lane_invasion")

    # x=profondità y=destra_sinistra z=altezza
    lane_location = carla.Transform(carla.Location())
    lane_sensor = world.spawn_actor(lane_bp, lane_location, attach_to=vehicle.vehicle_actor)

    lane_sensor.listen(lambda data: print(data.crossed_lane_markings[0].type))

    actors.append(vehicle.vehicle_actor)

    for _ in range(0, 2):
        controls = vehicle.get_random_action()
        print(controls)
        vehicle.vehicle_actor.apply_control(
            carla.VehicleControl(
                throttle=controls["throttle"],
                steer=controls["steer"],
                brake=controls["brake"],
                hand_brake=controls["handbrake"],
                reverse=controls["reverse"]
            ))
        time.sleep(5)

    cam_sensor.stop()
    lane_sensor.stop()

    for i, im in enumerate(l_images):
        # cv2.imwrite(f"images/image{i}.png",im)
        cv2.imshow("image", im)
        cv2.waitKey(15)

finally:
    for a in actors:
        a.destroy()
