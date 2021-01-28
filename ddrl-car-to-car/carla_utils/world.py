import carla
import random
from carla_utils import config
from carla_utils.actors import Vehicle


class World:

    def __init__(self, host=config.HOST, port=config.PORT):
        self.client = carla.Client(host, port)
        self.client.set_timeout(config.CLIENT_TIMEOUT)
        self.world = self.client.reload_world()  # self.client.get_world()
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.blueprint_library = self.world.get_blueprint_library()
        self.map_name = self.world.get_map().name

    def load_map(self, map_name="Town01", random_choice=True):
        self.map_name = random.choice(self.client.get_available_maps()) if random_choice else map_name
        self.world = self.client.load_world(self.map_name)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def create_vehicle(self, position, model=config.DEFAULT_VEHICLE_MODEL):
        vehicle_blueprint = self.blueprint_library.filter(model)[0]
        vehicle = self.world.spawn_actor(vehicle_blueprint, position)
        return Vehicle(vehicle_actor=vehicle)

    def attach_sensor_to_vehicle(self, vehicle, sensor):
        sensor_actor = self.world.spawn_actor(sensor.blueprint, sensor.location, attach_to=vehicle.vehicle_actor)
        sensor_actor.listen(lambda data: sensor.callback(data))
        sensor.sensor_actor = sensor_actor
        return sensor

    def attach_sensors_to_vehicle(self, vehicle, sensors=[]):
        result = []
        for sensor in sensors:
            result.append(self.attach_sensor_to_vehicle(vehicle, sensor))
        return result

    def get_map(self):
        return self.world.get_map()

    def get_turns_spawn_points_indexes(self):
        if self.map_name in config.TOWNS_TURNS_SP:
            return config.TOWNS_TURNS_SP[self.map_name]
        return self.get_spawn_points()

    def get_spawn_points(self):
        return self.get_map().get_spawn_points()
