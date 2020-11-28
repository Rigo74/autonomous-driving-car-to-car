import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import random
import time
import numpy as np
import tensorflow as tf
from threading import Thread
from tqdm import tqdm

from dqn_agent import DQNAgent
from dqn_parameters import *
from environment import CarlaEnvironment
from rewards import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def generate_model_name_appendix(max_reward, average_reward, min_reward, episode, epsilon):
    return f"{episode}ep_{epsilon}eps_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min"


trained_model_name = None  # "Cnn4Layers_1606463958_500ep____8.40max___-4.57avg__-30.80min"

if __name__ == '__main__':
    max_reward = average_reward = min_reward = 0
    epsilon = INITIAL_EPSILON
    # For stats
    ep_rewards = [MIN_REWARD]

    # For more repetitive results
    # random.seed(1)
    # np.random.seed(1)
    # tf.random.set_seed(1)

    # Create agent and carla_utils
    env = CarlaEnvironment(camera_config=(
        RGB_CAMERA_IM_WIDTH,
        RGB_CAMERA_IM_HEIGHT,
        RGB_CAMERA_FOV
    ))
    agent = DQNAgent(MODEL_NAME, env.get_number_of_actions())
    agent.load_model(trained_model_name)

    try:
        # Start training thread and wait for training to be initialized
        trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
        trainer_thread.start()
        while not agent.training_initialized:
            time.sleep(0.01)

        # Initialize predictions - forst prediction takes longer as of initialization that has to be done
        # It's better to do a first prediction then before we start iterating over episode steps
        agent.get_qs(np.ones((env.front_camera.im_height, env.front_camera.im_width, env.front_camera.channels)))

        # Iterate over episodes
        for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
            # try:

            # Update tensorboard step every episode
            agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0

            # Reset carla_utils and get initial state
            env.reset()
            # env.move_view_to_vehicle_position()
            current_state = env.get_current_state()
            # print("[DDQN] current state (rgb image)")
            # print(current_state)

            # Reset flag and start iterating until episode ends
            done = False
            episode_end = time.time() + SECONDS_PER_EPISODE

            step = 1
            start = time.time()
            # Play for given number of seconds only
            while not done:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, env.get_number_of_actions())
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1 / FPS)

                new_state, reward, done = env.step(action)

                if episode_end < time.time():
                    done = True

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1

            end = time.time()
            time_elapsed = end - start
            # print()
            # print(f"[STEPS_IN_EPISODE] {step}")
            # print(f"[ELAPSED_TIME] {time_elapsed}")
            # print(f"[STEP_AVG_TIME] {time_elapsed/step}")

            # End of episode - destroy agents
            env.destroy()

            agent.log_metrics(episode_reward, epsilon)

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY_X_EPISODES or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY_X_EPISODES:]) \
                                 / len(ep_rewards[-AGGREGATE_STATS_EVERY_X_EPISODES:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY_X_EPISODES:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY_X_EPISODES:])

                print(
                    f"<<< episode #{episode} >>> average_reward = {average_reward} | min_reward = {min_reward} | max_reward = {max_reward}")

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.save_model(generate_model_name_appendix(max_reward, average_reward, min_reward, episode,epsilon))

            # Decay epsilon
            if epsilon > FINAL_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(FINAL_EPSILON, epsilon)

        # Set termination flag for training thread and wait for it to finish
        agent.terminate = True
        trainer_thread.join()
        agent.save_model(generate_model_name_appendix(max_reward, average_reward, min_reward, agent.tensorboard.step,epsilon))
    except Exception as ex:
        print("[SEVERE] Exception raised: ")
        print(ex)
        env.destroy()
    finally:
        print("Terminated")
