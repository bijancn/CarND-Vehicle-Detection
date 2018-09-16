import sklearn
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers, callbacks
from model import *


SAVE_IMAGES = False
model = create_model()
model.summary()
model.add(Flatten())


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
  plt.savefig("output_images/training_data_overview.png")


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
if (SAVE_IMAGES):
    show_images(X_train)
    exit()

#  adam = optimizers.Adam(lr=0.00005)
#  model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
checkpoint = callbacks.ModelCheckpoint("weights.{epoch:02d}-{val_loss:.3f}.hdf5",
                          monitor='val_loss',
                          verbose=1,
                          save_best_only=True,
                          save_weights_only=False,
                          mode='auto',
                          period=1)
earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                    patience=5, verbose=1, mode='auto')

hist = model.fit(X_train, Y_train,
                 batch_size=128,
                 verbose=1,
                 epochs=20,
                 validation_data=(X_test, Y_test),
                 callbacks=[checkpoint, earlystop])

print(hist.history)
pickle.dump(hist.history, open( "history.pickle", "wb" ) )
model.save_weights('./model.h5')
plot_training(hist)
