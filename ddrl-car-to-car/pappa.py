import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from dqn_parameters import *
from models import Cnn64x3_Conv3D

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

model = Cnn64x3_Conv3D.create_model(10)

#images = np.ones(shape=(4, 120, 160, 3))
# y = np.array(np.ones(shape=12))

#print(model(np.array(images).reshape(-1, *images.shape)))

model.summary()

print("---------------------------------------------------------------------")

Cnn64x3.create_model(10).summary()
