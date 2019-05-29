import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import optimizers


def train_model(train_folder, val_size=.1, lr=.0001, batch_size=42, img_shape=(512, 512), input_shape=(512, 512, 1),
                epochs=50):
    """
    Train a model based on training data.
    :param train_folder: string, folder with learn data inside
    :param val_size: float, validation size
    :param lr: int, learning rate
    :param batch_size: int, size of a batch
    :param img_shape: tuple of int, shape of data images (x an y)
    :param input_shape: tuple of int, shape of data given to neural network
    :param epochs: int, number of epochs
    :return: trained model
    """

    model = create_model(input_shape)

    train_datagen = ImageDataGenerator(
        samplewise_std_normalization=True,
        validation_split=val_size,
        rescale=1. / 255,
    )

    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=img_shape,
        batch_size=batch_size,
        shuffle=True,
        subset="training",
        class_mode='categorical',
        color_mode='grayscale')

    if val_size > 0:
        validation_generator = train_datagen.flow_from_directory(
            train_folder,
            target_size=img_shape,
            batch_size=batch_size,
            shuffle=True,
            subset="validation",
            class_mode='categorical',
            color_mode='grayscale')

    n_classes = train_generator.num_classes

    model.add(Dense(n_classes, activation='softmax'))
    opt = optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint('model', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.0001)

    model.fit_generator(
        train_generator,
        steps_per_epoch=(len(train_generator.classes) // batch_size),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=(len(validation_generator.classes) // batch_size),
        verbose=1,
        callbacks=[checkpoint, reduce_lr],
    )

    return model


def create_model(input_shape):
    """
    Create a premade convolution 2D neural network with 8 convolution layers and 2 dense.
    :param input_shape: tuple if integers, shape of the input data
    :return: model
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(512, use_bias=True, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, use_bias=True, activation='relu'))
    model.add(Dropout(0.5))

    return model


def predict(test_folder, model=None, img_shape=(512, 512), predict_file='res.txt'):
    """
    Predict classification of test data on a given pretrained model
    :param test_folder: string, folder with test data
    :param model: Keras model, pretrained model
    :param img_shape: tuple of int, shape of input data
    :param predict_file: string, where are predictions saved
    """
    test_datagen = ImageDataGenerator(
        samplewise_std_normalization=True,
        rescale=1. / 255,
    )

    pred_generator = test_datagen.flow_from_directory(
        test_folder,
        target_size=img_shape,
        shuffle=False,
        color_mode='grayscale'
    )

    pred_generator.reset()

    pred = model.predict_generator(pred_generator)
    res = pred.argmax(axis=-1)
    print(res)

    np.savetxt(predict_file, res, fmt='%i')


if __name__ == '__main__':
    model = train_model('./data')
    predict('./data', model=model)
