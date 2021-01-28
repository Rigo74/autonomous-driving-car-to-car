import os
import time
import carla

from carla_utils.world import World

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

world = World()
spawn_points = world.get_turns_spawn_points_indexes()
print(world.map_name)
print(f"count: {len(spawn_points)}")
print(spawn_points)
'''
for i, sp in enumerate(spawn_points):
    vehicle = world.create_vehicle(position=sp)
    spectator = world.world.get_spectator()
    transform = vehicle.vehicle_actor.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
    print(i)
    print("---------------------------")
    time.sleep(3)
    vehicle.vehicle_actor.destroy()
'''
