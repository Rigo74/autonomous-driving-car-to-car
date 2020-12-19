from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Activation, AveragePooling2D, Flatten, Input, \
    Concatenate
from keras.optimizers import Adam
from keras.models import Model, Sequential

from config import *


def create_model_from_name(model_name, number_of_actions):
    if model_name == XceptionModel.get_model_name():
        return XceptionModel.create_model(number_of_actions)
    elif model_name == Cnn4Layers.get_model_name():
        return Cnn4Layers.create_model(number_of_actions)
    elif model_name == Cnn4LayersWithSpeed.get_model_name():
        return Cnn4LayersWithSpeed.create_model(number_of_actions)
    elif model_name == Cnn64x3.get_model_name():
        return Cnn64x3.create_model(number_of_actions)
    else:
        raise ValueError(f"The chosen model ({model_name}) is not available")


class XceptionModel(object):

    @staticmethod
    def get_model_name():
        return "Xception"

    @staticmethod
    def create_model(number_of_actions):
        base_model = Xception(
            weights=None,
            include_top=False,
            input_shape=IMG_DIMENSION
        )

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(number_of_actions, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model


class Cnn4Layers(object):

    @staticmethod
    def get_model_name():
        return "Cnn4Layers"

    @staticmethod
    def create_image_model():
        model = Sequential()

        model.add(Conv2D(64, (5, 5), input_shape=IMG_DIMENSION, padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(128, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

        model.add(Flatten())
        return model

    @staticmethod
    def create_model(number_of_actions):
        model = Cnn4Layers.create_image_model()
        model.add(Dense(128, activation="relu"))
        model.add(Dense(number_of_actions, activation="linear"))
        model.compile(loss="huber_loss", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model


class Cnn4LayersWithSpeed(object):

    @staticmethod
    def get_model_name():
        return "Cnn4LayersWithSpeed"

    @staticmethod
    def create_model(number_of_actions):
        image_model = Cnn4Layers.create_image_model()
        inputs = [image_model.input]
        speed_input = Input(shape=(1,), name='speed_input')
        inputs.append(speed_input)

        x = Concatenate()([image_model.output, speed_input])
        x = Dense(128, activation="relu")(x)
        predictions = Dense(number_of_actions, activation='linear')(x)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss="huber_loss", optimizer=Adam(lr=0.001), metrics=["accuracy"])

        return model


class Cnn64x3(object):

    @staticmethod
    def get_model_name():
        return "Cnn64x3"

    @staticmethod
    def create_model(number_of_actions):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=IMG_DIMENSION, padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Flatten())

        model.add(Dense(128, activation="relu"))

        model.add(Dense(number_of_actions, activation="linear"))

        model.compile(loss="huber_loss", optimizer=Adam(lr=0.001), metrics=["accuracy"])

        return model
