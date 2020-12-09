import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout


def cnn_binary(wm_dim):
    np.random.seed(7)
    tf.random.set_seed(7)
    n_cls = 2
    K.clear_session()
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(wm_dim, wm_dim, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_cls, activation='softmax'))

    model.compile('nadam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def cnn_multi(wm_dim):
    np.random.seed(7)
    tf.random.set_seed(7)
    n_cls = 8
    K.clear_session()
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(wm_dim, wm_dim, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_cls, activation='softmax'))
    
    model.compile('nadam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model
