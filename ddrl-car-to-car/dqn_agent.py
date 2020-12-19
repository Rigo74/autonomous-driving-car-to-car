import random
import time
import threading
import keras
import os
import numpy as np
from collections import deque
from keras.callbacks import TensorBoard
import tensorflow as tf
# from sys import getsizeof

from dqn_parameters import *
from config import *
import models

MODELS_FOLDER = "models"
LOGS_FOLDER = "logs"

# Custom Metrics
REWARD = "reward"
EPSILON = "epsilon"


class DQNAgent:
    def __init__(self, model_name, number_of_actions):
        self.model_lock = threading.Lock()
        self.number_of_actions = number_of_actions
        self.model = models.create_model_from_name(model_name, number_of_actions=number_of_actions)
        self.target_model = models.create_model_from_name(model_name, number_of_actions=number_of_actions)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.identifier = int(time.time())
        self.model_name = f"{MODEL_NAME}_{self.identifier}"

        model_logs_folder = f"{LOGS_FOLDER}/{self.model_name}"

        file_writer = tf.summary.create_file_writer(model_logs_folder + "/metrics")
        file_writer.set_as_default()

        self.tensorboard = TensorBoard(log_dir=model_logs_folder)
        self.target_update_counter = 0

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

        if not os.path.isdir(MODELS_FOLDER):
            os.makedirs(MODELS_FOLDER)

    def log_metrics(self, reward, epsilon):
        tf.summary.scalar(REWARD, data=reward, step=self.tensorboard.step)
        tf.summary.scalar(EPSILON, data=epsilon, step=self.tensorboard.step)

    def load_model(self, model_name):
        model_path = f"{MODELS_FOLDER}/{model_name}"
        if model_name is not None:
            self.model = keras.models.load_model(model_path)
            self.target_model = keras.models.load_model(model_path)

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        # print(f"[REPLAY_MEMORY] len: {len(self.replay_memory)} size: {getsizeof(self.replay_memory)}")
        # print(f"[REPLAY_MEMORY] len: {len(self.replay_memory)}")
        self.do_synchronized(lambda: self.replay_memory.append(transition))

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        mini_batch = random.sample(self.replay_memory, MINI_BATCH_SIZE)

        current_states = np.array([transition[0] for transition in mini_batch]).astype(np.float32) / 255.0
        current_qs_list = self.do_synchronized(lambda: self.model.predict(current_states, PREDICTION_BATCH_SIZE))

        new_current_states = np.array([transition[3] for transition in mini_batch]).astype(np.float32) / 255.0
        future_qs_list = self.do_synchronized(lambda: self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE))

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(mini_batch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        X = np.array(X).astype(np.float32) / 255.0
        y = np.array(y).astype(np.float32)

        self.do_synchronized(lambda: self.model.fit(
            X,
            y,
            batch_size=TRAINING_BATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if log_this_step else None
        ))

        self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY_X_STEPS:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        prediction_input = np.array(state).astype(np.float32).reshape(-1, *state.shape) / 255.0
        return self.do_synchronized(lambda: self.model.predict(prediction_input))[0]

    def save_model(self, name_appendix):
        try:
            self.do_synchronized(lambda: self.model.save(f'{MODELS_FOLDER}/{self.model_name}_{name_appendix}'))
        except Exception as ex:
            print("[SEVERE] Exception raised while saving: ")
            print(ex)

    def train_in_loop(self):
        input_size = (1, RGB_CAMERA_IM_HEIGHT, RGB_CAMERA_IM_WIDTH, RGBCamera.get_number_of_channels())
        X = np.random.uniform(size=input_size).astype(np.float32)
        y = np.random.uniform(size=(1, self.number_of_actions)).astype(np.float32)
        self.do_synchronized(lambda: self.model.fit(X, y, verbose=False, batch_size=1))

        self.training_initialized = True

        while not self.terminate:
            self.train()
            time.sleep(0.01)

    def do_synchronized(self, function_to_execute):
        self.model_lock.acquire(True)
        result = function_to_execute()
        self.model_lock.release()
        return result
