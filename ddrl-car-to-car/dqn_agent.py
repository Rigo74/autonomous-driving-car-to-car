import random
import time
import threading
import keras
import numpy as np
from collections import deque
from tensorflow.keras.callbacks import TensorBoard

from dqn_parameters import *
from config import *
import models


class DQNAgent:
    def __init__(self, model_name, number_of_actions):
        self.model_lock = threading.Lock()
        self.number_of_actions = number_of_actions
        self.model = models.create_model_from_name(model_name, number_of_actions=number_of_actions)
        self.target_model = models.create_model_from_name(model_name, number_of_actions=number_of_actions)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = TensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.target_model = keras.models.load_model(model_path)

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        mini_batch = random.sample(self.replay_memory, MINI_BATCH_SIZE)

        current_states = np.array([transition[0] for transition in mini_batch]) / 255.0
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in mini_batch]) / 255.0
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

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
            self.last_log_episode = self.tensorboard.step

        self.model.fit(
            np.array(X) / 255.0,
            np.array(y),
            batch_size=TRAINING_BATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if log_this_step else None
        )

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255.0)[0]

    def save_model(self, name):
        try:
            self.model_lock.acquire(True)
            self.model.save(f'models/{name}')
            self.model_lock.release()
        except Exception as ex:
            print("[SEVERE] Exception raised while saving: ")
            print(ex)

    def train_in_loop(self):
        input_size = (1, RGB_CAMERA_IM_HEIGHT, RGB_CAMERA_IM_WIDTH, RGBCamera.get_number_of_channels())
        X = np.random.uniform(size=input_size).astype(np.float32)
        y = np.random.uniform(size=(1, self.number_of_actions)).astype(np.float32)
        self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while not self.terminate:
            self.model_lock.acquire(True)
            self.train()
            self.model_lock.release()
            time.sleep(0.01)
