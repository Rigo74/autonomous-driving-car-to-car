import gc
import os
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import random
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from guppy import hpy

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
    max_reward = average_reward = min_reward = 0
    epsilon = INITIAL_EPSILON

    h = hpy()

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
        agent.get_qs(np.ones((env.front_camera.im_height, env.front_camera.im_width, env.front_camera.channels)))

        for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
            agent.tensorboard.step = episode
            episode_reward = 0

            env.reset()
            env.move_view_to_vehicle_position()
            current_state = env.get_current_state()

            done = False

            step = 0
            start = time.time()

            episode_losses = []

            while not done:

                step_start_time = time.time()

                if np.random.random() > epsilon:
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    action = np.random.randint(0, env.get_number_of_actions())

                new_state, reward, done = env.step(action)
                episode_reward += reward
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1
                if step >= STEPS_PER_EPISODE:  # episode_end < time.time():
                    done = True

                loss = agent.train()
                if loss is not None:
                    episode_losses.extend(loss)

                time.sleep(STEP_TIME_SECONDS - (time.time() - step_start_time))

            end = time.time()
            time_elapsed = end - start

            # print(f"[EPISODE] {episode} [SECONDS] {time_elapsed} [STEPS] {step} [REWARD] {episode_reward}")

            env.destroy()

            last_x_episodes_rewards.append(episode_reward)
            episode_average_loss = np.mean(episode_losses)
            last_x_episodes_losses.append(episode_average_loss)
            average_reward = np.mean(last_x_episodes_rewards)
            average_loss = np.mean(last_x_episodes_losses)

            agent.log_metrics(episode_reward, epsilon, episode_average_loss, average_reward, average_loss)

            if not episode % AGGREGATE_STATS_EVERY_X_EPISODES or episode == 1:
                min_reward = min(last_x_episodes_rewards)
                max_reward = max(last_x_episodes_rewards)

                print(
                    f"<<< episode #{episode} >>> average_reward = {average_reward} | min_reward = {min_reward} | max_reward = {max_reward}")

                if min_reward >= MIN_REWARD:
                    agent.save_model(generate_model_name_appendix(max_reward, average_reward, min_reward, episode, epsilon))

            if epsilon > FINAL_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(FINAL_EPSILON, epsilon)

            gc.collect()

            if not (episode % PRINT_X_HEAP_VARIABLES_EVERY_Y_EPISODES):
                print(h.heap()[:PRINT_X_HEAP_VARIABLES].all)

        agent.terminate = True
        agent.save_model(generate_model_name_appendix(max_reward, average_reward, min_reward, agent.tensorboard.step, epsilon))
    except Exception as ex:
        print("[SEVERE] Exception raised: ")
        print(ex)
        env.destroy()
    finally:
        print("Terminated")
