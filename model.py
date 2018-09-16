import glob
import skimage.io
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import *


def load_data():
  cars = glob.glob("./data/vehicles/*/*.png")
  non_cars = glob.glob("./data/non-vehicles/*/*.png")
  #  cars = cars[0:int(len(cars)/100)]
  #  non_cars = non_cars[0:int(len(non_cars)/100)]
  X = np.array(list(map(lambda x: skimage.io.imread(x), cars + non_cars)))
  Y = np.concatenate([np.ones(len(cars)), np.zeros(len(non_cars))])
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
  uniq_train, training_counts = np.unique(Y_train, return_counts = True)
  uniq_test, test_counts = np.unique(Y_test, return_counts = True)
  print(uniq_train, uniq_test)
  print('X_train shape:', X_train.shape)
  print(X_train.shape[0], 'train samples')
  print(X_test.shape[0], 'test samples')
  print(training_counts[0], 'noncars to', training_counts[1], 'cars in training set (1 :',
        1.0 * training_counts[0] / training_counts[1], ')')
  print(test_counts[0], 'noncars to', test_counts[1], 'cars in test set (1 :',
        1.0 * test_counts[0] / test_counts[1], ')')
  return X_train, X_test, Y_train, Y_test


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
