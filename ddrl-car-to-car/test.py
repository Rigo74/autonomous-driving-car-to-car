import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import time
import numpy as np
import tensorflow as tf
from environment import CarlaEnvironment
from dqn_parameters import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)

model_name = "models/Cnn64x3_1611342485/Cnn64x3_1611342485_3300ep_0.1eps__154.30max___39.31avg___-1.00min"

if __name__ == '__main__':

    # Create agent and carla_utils
    env = CarlaEnvironment(camera_config=(
        RGB_CAMERA_IM_WIDTH,
        RGB_CAMERA_IM_HEIGHT,
        RGB_CAMERA_FOV
    ))

    try:
        env.reset()
        model = keras.models.load_model(model_name)

        env.move_view_to_vehicle_position()
        # env.vehicle.move(throttle=1.0)

        # for i in range(0, 10):
        while True:
            step_start_time = time.time()
            state = env.get_current_state()
            choices = model(np.array(state).reshape(-1, *state.shape))
            choice = np.argmax(choices)
            action = env.do_action(choice)
            print(f"{choice} -> {action}")
            step_elapsed_time = time.time() - step_start_time
            print(step_elapsed_time)
            if STEP_TIME_SECONDS > step_elapsed_time:
                time.sleep(STEP_TIME_SECONDS - step_elapsed_time)

    except Exception as ex:
        print("[SEVERE] Exception raised: ")
        print(ex)
        env.destroy()
    finally:
        print("Terminated")
        env.destroy()
