import os
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dqn_agent import DQNAgent
from dqn_parameters import *
from environment import CarlaEnvironment
from rewards import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def generate_model_name_appendix(max_reward, average_reward, min_reward, episode, epsilon):
    return f"{episode}ep_{epsilon}eps_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min"


trained_model_name = None  # "Cnn4Layers_1607586908_18200ep_0.1eps____3.00max___-0.61avg___-3.00min"

if __name__ == '__main__':

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    max_reward = average_reward = average_loss = min_reward = 0
    epsilon = INITIAL_EPSILON

    step_times = []

    last_x_episodes_rewards = deque(maxlen=AGGREGATE_STATS_EVERY_X_EPISODES)
    last_x_episodes_rewards.append(MIN_REWARD)

    last_x_episodes_losses = deque(maxlen=AGGREGATE_STATS_EVERY_X_EPISODES)

    # Create agent and carla_utils
    env = CarlaEnvironment(camera_config=(
        RGB_CAMERA_IM_WIDTH,
        RGB_CAMERA_IM_HEIGHT,
        RGB_CAMERA_FOV
    ))
    agent = DQNAgent(MODEL_NAME, env.get_number_of_actions())
    agent.load_model(trained_model_name)
    agent.initialize_training_variables()

    try:
        # Initialize predictions - first prediction takes longer as of initialization that has to be done
        # It's better to do a first prediction then before we start iterating over episode steps
        pippo = []
        for h in range(0,env.front_camera.im_height):
            a = []
            for w in range(0, env.front_camera.im_width):
                b = []
                for c in range(0, env.front_camera.channels):
                    b.append(1)
                a.append(b)
            pippo.append(a)

        pippo_np = np.ones((
            env.front_camera.im_height,
            env.front_camera.im_width,
            env.front_camera.channels
        ))

        banana = agent.get_qs(pippo_np)

        print(banana)

    except Exception as ex:
        print("[SEVERE] Exception raised: ")
        print(ex)
        env.destroy()
    finally:
        print("Terminated")
