import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

IM_WIDTH = 640
IM_HEIGHT = 480

'''
# RGB image processing
input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
conv1 = Conv2D(32, kernel_size=5)(input_layer)
# conv2 = Conv2D(32, kernel_size=3, strides=1, input_shape=(98, 48, 32))(conv1)
# conv3 = Conv2D(64, kernel_size=3, strides=2, input_shape=(96, 46, 32))(conv2)
# conv4 = Conv2D(64, kernel_size=3, strides=1, input_shape=(47, 22, 64))(conv3)
# flatten = Flatten()(conv1)
dense = Dense(1)(conv1)
model = Model(input_layer, dense)
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

model = Sequential()
model.add(Input((IM_HEIGHT, IM_WIDTH, 3)))
model.add(Conv2D(32, kernel_size=5))
# model.add(Conv2D(32, kernel_size=3, strides=1, input_shape=(48, 98, 32)))
# model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(46, 96, 32)))
# model.add(Conv2D(64, kernel_size=3, strides=1, input_shape=(22, 47, 64)))
model.compile(optimizer='adam')
'''
num_actions = 4

# Network defined by the Deepmind paper
inputs = layers.Input(shape=(IM_HEIGHT, IM_WIDTH, 3))

# Convolutions on the frames on the screen
layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

layer4 = layers.Flatten()(layer3)

layer5 = layers.Dense(64, activation='relu')(layer4)
action = layers.Dense(num_actions)(layer5)

model = keras.Model(inputs=inputs, outputs=action)
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


images = []
for i in range(0, 200):
    image = cv2.imread(f"images/image{i}.png")
    image = image[:, :, :3]
    images.append(image)


images = np.array(images) / 255.0

labels = []

for i in range(0, 200):
    if i < 25:
        labels.append([1])
    elif i < 50:
        labels.append([2])
    elif i < 25:
        labels.append([3])
    else:
        labels.append([4])

labels = np.array(labels)

train_images = images[0:170]
train_labels = labels[0:170]

test_images = images[170:]
test_labels = labels[170:]

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)