from keras.models import Sequential
from keras.layers import *

def create_model(input_shape=(64,64,3)):
  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape, output_shape=input_shape))
  model.add(Convolution2D(128, 3, 3, activation='relu', name='conv1',
            input_shape=input_shape, border_mode="same"))
  model.add(Dropout(0.5))
  model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2',border_mode="same"))
  model.add(Dropout(0.5))
  model.add(Convolution2D(128, 3, 3, activation='relu', name='conv3',border_mode="same"))
  model.add(MaxPooling2D(pool_size=(8, 8)))
  model.add(Dropout(0.5))
  model.add(Convolution2D(128, 8, 8, activation="relu", name="dense1"))
  model.add(Dropout(0.5))
  model.add(Convolution2D(1, 1, 1, name="dense2", activation="tanh"))
  return model

model = create_model()
model.summary()
model.add(Flatten())
