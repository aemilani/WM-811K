import os
import datetime
import numpy as np
import tensorflow.keras.backend as K
import datset as ds
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


os.mkdir('grid_chp/')

wm_dim = 64
x_train, y_train, x_valid, y_valid, x_test, y_test = \
    ds.dataset(include_nonpattern=False, wm_dim=wm_dim)

hidden_1 = [16, 32]
hidden_2 = [32, 64]
hidden_3 = [32, 64]
hidden_4 = [64, 128]
dense = [128, 256]
batch = [512, 1024]
opt = ['adam', 'nadam']

dic_loss = {}
dic_acc = {}

for h1 in hidden_1:
    for h2 in hidden_2:
        for h3 in hidden_3:
            for h4 in hidden_4:
                for d in dense:
                    np.random.seed(13)
                    time = str(datetime.datetime.now()).split()[1]
                    
                    K.clear_session()
                    model = Sequential()
                    model.add(Conv2D(h1, (3, 3), padding='same', activation='relu', input_shape=(wm_dim, wm_dim, 3)))
                    model.add(MaxPooling2D())
                    model.add(Conv2D(h2, (3, 3), padding='same', activation='relu'))
                    model.add(MaxPooling2D())
                    model.add(Conv2D(h3, (3, 3), padding='same', activation='relu'))
                    model.add(MaxPooling2D())
                    model.add(Conv2D(h4, (3, 3), padding='same', activation='relu'))
                    model.add(MaxPooling2D())
                    model.add(Flatten())
                    model.add(Dropout(0.25))
                    model.add(Dense(d, activation='relu'))
                    model.add(Dropout(0.5))
                    model.add(Dense(8, activation='softmax'))
                    
                    model.compile('nadam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
                    
                    ea = EarlyStopping(patience=5)
                    cp = ModelCheckpoint('grid_chp/{}.h5'.format(time), save_best_only=True)
                    cb = [ea, cp]
                    
                    model.fit(x_train, y_train, batch_size=1024, epochs=30,
                                        validation_data=(x_valid, y_valid), callbacks=cb)
                    
                    model = load_model('grid_chp/{}.h5'.format(time))
                    
                    score = model.evaluate(x_test, y_test, verbose=0)
                    test_loss = score[0]
                    test_acc = score[1]
                    dic_loss['hidden_1: {}, hidden_2: {}, hidden_3: {}, hidden_4: {}, dense: {}' \
                        .format(h1, h2, h3, h4, d)] = test_loss
                    dic_acc['hidden_1: {}, hidden_2: {}, hidden_3: {}, hidden_4: {}, dense: {}' \
                        .format(h1, h2, h3, h4, d)] = test_acc
                    
print('Best setting (loss):', min(dic_loss))
print('Best score (loss):', dic_loss[min(dic_loss)])
print('Best setting (acc):', max(dic_acc))
print('Best score (acc):', dic_acc[max(dic_acc)])