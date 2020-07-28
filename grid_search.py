import os
import datetime
import numpy as np
import tensorflow.keras.backend as K
import dataset as ds
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

result = []

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
                    dic = {'hidden_1': h1, 'hidden_2': h2, 'hidden_3': h3,
                           'hidden_4': h4, 'dense': d}
                    result.append((dic, test_loss, test_acc))

result.sort(key=lambda x: x[1])
print('Best setting (loss):', result[0][0])
print('Best score (loss):', result[0][1], result[0][2])

result.sort(key=lambda x: x[2])
print('Best setting (acc):', result[-1][0])
print('Best score (acc):', result[-1][1], result[-1][2])