import time
import tensorflow.keras as keras
import os
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

from tensorflow.keras import losses

from replay_memory import *
from dqn_parameters import *
from config import *
import models

MODELS_FOLDER = "models"
LOGS_FOLDER = "logs"


def reshape(state, add_external_array=True):
    state_np = np.asarray(state)
    shape = state_np.shape
    temp = (shape[0] * shape[1], shape[2], shape[3])
    return state_np.reshape((-1, *temp) if add_external_array else temp)


class DQNAgent:
    def __init__(self, model_name, number_of_actions):
        self.number_of_actions = number_of_actions
        self.behaviour_model = models.create_model_from_name(model_name, number_of_actions=number_of_actions)
        self.target_model = models.create_model_from_name(model_name, number_of_actions=number_of_actions)
        self.target_model.set_weights(self.behaviour_model.get_weights())

        self.replay_memory = ReplayMemory(max_len=REPLAY_MEMORY_SIZE, min_len=MIN_REPLAY_MEMORY_SIZE)

        self.identifier = int(time.time())
        self.model_name = f"{MODEL_NAME}_{self.identifier}"

        model_logs_folder = f"{LOGS_FOLDER}/{self.model_name}"

        file_writer = tf.summary.create_file_writer(model_logs_folder + "/metrics")
        file_writer.set_as_default()

        self.tensorboard = TensorBoard(log_dir=model_logs_folder)
        self.target_update_counter = 0

        self.model_folder = f"{MODELS_FOLDER}/{self.model_name}"

        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)

        self.behaviour_model.summary()

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            tf.summary.scalar(key, data=value, step=self.tensorboard.step)

    def load_model(self, model_name):
        model_path = f"{MODELS_FOLDER}/{model_name}"
        if model_name is not None:
            self.behaviour_model = keras.models.load_model(model_path)
            self.target_model = keras.models.load_model(model_path)
            self.behaviour_model.summary()

    def update_replay_memory(self, old_state, action, reward, new_state, done):
        self.replay_memory.add_transition(Transition(old_state, new_state, action, reward, done))

    def train(self):

        if not self.replay_memory.has_enough_values():
            return None

        batch = self.replay_memory.get_random_samples(MINI_BATCH_SIZE)

        old_states = np.asarray([reshape(sample.old_state, False) for sample in batch])
        new_states = np.asarray([reshape(sample.new_state, False) for sample in batch])
        actions = np.asarray([sample.action for sample in batch])
        rewards = np.asarray([sample.reward for sample in batch])
        is_done = np.asarray([sample.is_done for sample in batch])

        q_new_state = np.max(self.target_predict(new_states), axis=1)
        target_q = rewards + (DISCOUNT * q_new_state * (1 - is_done))
        one_hot_actions = keras.utils.to_categorical(actions, self.number_of_actions)

        loss = self.gradient_train(old_states, target_q, one_hot_actions)

        self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY_X_STEPS:
            self.behaviour_model.set_weights(self.target_model.get_weights())
            self.target_update_counter = 0

        return loss

    @tf.function
    def target_predict(self, state):
        # return tf.numpy_function(lambda s: self.target_model(s), [state], tf.float32)
        return self.target_model(state)

    @tf.function
    def behaviour_predict(self, state):
        # return tf.numpy_function(lambda s: self.model(s), [state], tf.float32)
        return self.behaviour_model(state)

    '''
    def my_numpy_func(self, state):
        # x will be a numpy array with the contents of the input to the
        # tf.function
        return self.model.predict(state)

    @tf.function#(input_signature=[tf.TensorSpec(None, tf.float32)])
    def tf_function(self, state):
        y = tf.numpy_function(self.my_numpy_func, [state], tf.float32)
        return y
    '''

    @tf.function
    def gradient_train(self, old_states, target_q, one_hot_actions):
        with tf.GradientTape() as tape:
            q_values = self.target_model(old_states)
            current_q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
            loss = losses.Huber()(target_q, current_q)

        variables = self.target_model.trainable_variables
        gradients = tape.gradient(loss, variables)
        zipped = zip(gradients, variables)
        self.target_model.optimizer.apply_gradients(zipped)

        return loss

    def get_qs(self, state):
        return self.behaviour_predict(reshape(np.array(state)))

    def save_model(self, name_appendix):
        try:
            self.target_model.save(f'{self.model_folder}/{self.model_name}_{name_appendix}')
        except Exception as ex:
            print("[SEVERE] Exception raised while saving: ")
            print(ex)

    def initialize_training_variables(self):
        input_size = (1, RGB_CAMERA_IM_HEIGHT, RGB_CAMERA_IM_WIDTH, RGBCamera.get_number_of_channels())
        X = np.random.uniform(size=input_size).astype(np.float32)
        y = np.random.uniform(size=(1, self.number_of_actions)).astype(np.float32)
        self.behaviour_model.fit(X, y, verbose=False, batch_size=1)
