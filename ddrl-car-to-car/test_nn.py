from keras.layers import Input, Conv2D, Flatten, Dense
from keras import Sequential
from keras.models import Model
import cv2
import numpy as np

IM_WIDTH = 640
IM_HEIGHT = 480

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


'''
model = Sequential()
model.add(Input((IM_HEIGHT, IM_WIDTH, 3)))
model.add(Conv2D(32, kernel_size=5))
# model.add(Conv2D(32, kernel_size=3, strides=1, input_shape=(48, 98, 32)))
# model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(46, 96, 32)))
# model.add(Conv2D(64, kernel_size=3, strides=1, input_shape=(22, 47, 64)))
model.compile(optimizer='adam')
'''

images = []
for i in range(0, 10):
    image = cv2.imread(f"images/image{i}.png")
    image = image[:, :, :3]
    # print(image.shape)
    images.append(image)
    # cv2.imshow("image", image)
    # cv2.waitKey(15)

y = []

for j in range(0, 10):
    t = []
    for i in range(0, IM_HEIGHT):
        t.append(np.ones(IM_WIDTH))
    y.append(np.array(t))

y = np.array(y)

model.fit(images, y)
