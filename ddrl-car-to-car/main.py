import os
import random
import time
import numpy as np
import tensorflow as tf
from threading import Thread
from tqdm import tqdm

from environment import CarlaEnvironment
from dqn_agent import DQNAgent
from config import *
from dqn_parameters import *
from rewards import *

if __name__ == '__main__':
    epsilon = INITIAL_EPSILON
    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and carla_utils
    agent = DQNAgent()
    env = CarlaEnvironment(camera_config=(
        RGB_CAMERA_IM_WIDTH,
        RGB_CAMERA_IM_HEIGHT,
        RGB_CAMERA_FOV
    ))
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

            env.collision_hist = []

            # Update tensorboard step every episode
            agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset carla_utils and get initial state
            env.reset()
            env.move_view_to_vehicle_position()
            current_state = env.get_current_state()
            # print("[DDQN] current state (rgb image)")
            # print(current_state)

            # Reset flag and start iterating until episode ends
            done = False
            episode_end = time.time() + SECONDS_PER_EPISODE

            # Play for given number of seconds only
            while not done:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, 3)
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1 / FPS)

                new_state, reward, done, _ = env.step(action)

                if episode_end < time.time():
                    done = True

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1

            # End of episode - destroy agents
            env.destroy()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY_X_EPISODES or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY_X_EPISODES:]) / len(
                    ep_rewards[-AGGREGATE_STATS_EVERY_X_EPISODES:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY_X_EPISODES:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY_X_EPISODES:])

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(
                        f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Decay epsilon
            if epsilon > FINAL_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(FINAL_EPSILON, epsilon)

        # Set termination flag for training thread and wait for it to finish
        agent.terminate = True
        trainer_thread.join()
        agent.model.save(
            f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
    except Exception:
        env.destroy()
    finally:
        print("Terminated")
