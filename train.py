import sklearn
import glob
import pickle
import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
#  from keras.layers import *
from keras import optimizers, callbacks
from sklearn.model_selection import train_test_split
from model import *


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


def show_images(X_train):
  fig = plt.figure(figsize=(12, 6))
  for i in range(0, 30):
    number = np.random.randint(0, len(X_train))
    axis = fig.add_subplot(3, 10, i+1)
    axis.set_xlabel(Y_train[number])
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    axis.imshow(X_train[number])
  fig.tight_layout()
  plt.show()


def plot_training(hist):
  plt.plot(hist.history['acc'])
  plt.plot(hist.history['val_acc'])
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Training', 'Test'], loc='upper left')
  plt.savefig('hist_accuracy.png')
  plt.gcf().clear()
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Training', 'Test'], loc='upper left')
  plt.savefig('hist_loss.png')


X_train, X_test, Y_train, Y_test = load_data()
adam = optimizers.Adam(lr=0.0001)
model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy'])
checkpoint = callbacks.ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                          monitor='val_loss',
                          verbose=1,
                          save_best_only=True,
                          save_weights_only=False,
                          mode='auto',
                          period=1)
earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                    patience=3, verbose=1, mode='auto')

hist = model.fit(X_train, Y_train,
                 batch_size=256,
                 verbose=1,
                 epochs=20,
                 validation_data=(X_test, Y_test),
                 callbacks=[checkpoint, earlystop])

print(hist.history)
plot_training(hist)
model.save_weights('./model.h5')
