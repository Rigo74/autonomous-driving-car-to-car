import carla
from carla_utils import config
from carla_utils.actors import Vehicle


class World:

    def __init__(self, host=config.HOST, port=config.PORT):
        self.client = carla.Client(host, port)
        self.client.set_timeout(config.CLIENT_TIMEOUT)
        self.world = self.client.reload_world()  # self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

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
