from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout


def cnn(wm_dim, include_nonpattern=False, **kwargs):
    if include_nonpattern:
        n_cls = 9
    else:
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
