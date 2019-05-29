import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils

from keras.datasets import mnist

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import os
from os import listdir, path

if __name__ == '__main__':
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    """
    learn_folders = ['data/yeast_images/train/0-CYTOPLASM', 'data/yeast_images/train/1-NUCLEUS', 'data/yeast_images/train/2-ER', 'data/yeast_images/train/3-MITOCHONDRIA']
    test_folders = ['data/yeast_images/test']

    # read data
    X_train = list()
    y_train = list()
    for i,folder in enumerate(learn_folders):
        for file in listdir(folder):
            p = path.join(folder, file)
            X_train.append(np.array(mpimg.imread(p), dtype='float'))
            y_train.append(i)
            #imgplot = plt.imshow(mpimg.imread(p))
            #plt.show()
    X_train = np.array(X_train)
    X_train /= 255
    n_classes = len(np.unique(y_train))
    im_size = 512


    X_test = list()
    for i, folder in enumerate(test_folders):
        for file in listdir(folder):
            p = path.join(folder, file)
            X_test.append(np.array(mpimg.imread(p), dtype='float'))
    X_test = np.array(X_test)
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, n_classes)

    model = Sequential()
    model.add(Convolution2D(32, 3, activation='relu', input_shape=(512, 512, 3), data_format='channels_last'))

    model.add(Convolution2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=128, nb_epoch=10, verbose=1)
    """
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model)
    model.save(model_path)
    """

    res = model.predict_classes(X_test)
    print(res)
    np.savetxt('res.txt', res, fmt='%i')
