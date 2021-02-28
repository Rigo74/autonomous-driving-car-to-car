import os
import time
import carla
import numpy as np

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
spawn_points = world.get_spawn_points()
print(world.map_name)
print(f"count: {len(spawn_points)}")
print(spawn_points)

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

c1 = np.ones((4, 3, 120, 160))

print(c1.reshape((12, 120, 160)).shape)

c2 = np.ones((4, 120, 160, 3))
c3 = np.ones((120, 160, 12))

for a4 in c2:
    for i, a120 in enumerate(a4):
        for j, a160 in enumerate(a120):
            for pantofola, v in enumerate(a160):
                c3[i][j][pantofola] = v

print(np.array(c3).shape)

print(c2.reshape())

# print(c1.reshape((12, 120, 160)).shape)



# 120 * 160 * 12

# 5 6 3

a = [
    [
        [
            [1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]
        ],
        [
            [1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]
        ],
        [
            [1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]
        ],
        [
            [1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]
        ],
        [
            [1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]
        ]
    ],
    [
        [
            [1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]
        ],
        [
            [1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]
        ],
        [
            [1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]
        ],
        [
            [1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]
        ],
        [
            [1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]
        ]
    ]
]


pantofola726 = [
    [
        [
            [1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]
        ],
        [
            [1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]
        ],
        [
            [1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]
        ],
        [
            [1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]
        ],
        [
            [1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]
        ]
    ]
]
# 3 5 6

b = [
    [
        [1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]
    ],
    [
        [1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]
    ],
    [
        [1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]
    ],
]
'''