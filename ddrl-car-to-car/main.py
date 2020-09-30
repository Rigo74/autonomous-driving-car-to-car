import carla
import time
from Vehicle import Vehicle


# CLIENT_ADDRESS = "localhost"
# CLIENT_PORT = 2000


def create_client(host="localhost", port=2000):
    _client = carla.Client(host, port)
    _client.set_timeout(10.0)
    _world = _client.get_world()
    # _world = _client.reload_world()
    # _world.set_weather(carla.WeatherParameters.ClearNoon)
    return _client, _world


def move_view_to_vehicle_position(world, vehicle):
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))


client, world = create_client()
vehicle = Vehicle("vehicle.audi.tt")

blueprint_library = world.get_blueprint_library()
vehicle_blueprint = blueprint_library.filter(vehicle.vehicle_model)[0]

spawn_point = world.get_map().get_spawn_points()[0]
vehicle.vehicle_actor = world.spawn_actor(vehicle_blueprint, spawn_point)
move_view_to_vehicle_position(world, vehicle.vehicle_actor)

for _ in range(0, 10):
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

vehicle.vehicle_actor.destroy()

'''
throttle=controls["throttle"],
steer=controls["steer"],
brake=controls["brake"],
hand_brake=controls["handbrake"],
reverse=controls["reverse"]
'''

'''
throttle=0.3,
steer=0.0,
brake=0.0,
hand_brake=False,
reverse=False
'''