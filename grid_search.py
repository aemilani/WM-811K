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


def build_cnn(n_h1, n_h2, n_h3, n_h4, n_d, opt, dim):
    K.clear_session()

    cnn = Sequential()
    cnn.add(Conv2D(n_h1, (3, 3), padding='same', activation='relu', input_shape=(dim, dim, 3)))
    cnn.add(MaxPooling2D())
    cnn.add(Conv2D(n_h2, (3, 3), padding='same', activation='relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Conv2D(n_h3, (3, 3), padding='same', activation='relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Conv2D(n_h4, (3, 3), padding='same', activation='relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Flatten())
    cnn.add(Dropout(0.25))
    cnn.add(Dense(n_d, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(8, activation='softmax'))
    cnn.compile(opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return cnn


def train_cnn(cnn, x_tr, y_tr, x_v, y_v, x_te, y_te, batch_size):
    if not os.path.exists('grid_chp/'):
        os.mkdir('grid_chp/')
    ea = EarlyStopping(patience=5)
    cp = ModelCheckpoint('grid_chp/chp.h5', save_best_only=True)
    cb = [ea, cp]
    cnn.fit(x_tr, y_tr, batch_size=batch_size, epochs=30,
            validation_data=(x_v, y_v), callbacks=cb, verbose=0)
    cnn = load_model('grid_chp/chp.h5')
    score = cnn.evaluate(x_te, y_te, verbose=0)
    return score


def grid_search_architecture(hidden_1, hidden_2, hidden_3, hidden_4, dense, dim):
    count_total = len(hidden_1) * len(hidden_2) * len(hidden_3) * len(hidden_4) * len(dense)
    result = []
    ctr = 1
    for h1 in hidden_1:
        for h2 in hidden_2:
            for h3 in hidden_3:
                for h4 in hidden_4:
                    for d in dense:
                        print('Architecture setting {} out of {} ...'.format(ctr, count_total))
                        model = build_cnn(n_h1=h1, n_h2=h2, n_h3=h3, n_h4=h4, n_d=d, opt='nadam', dim=dim)
                        test_score_1 = train_cnn(cnn=model, x_tr=x_train, y_tr=y_train, x_v=x_valid, y_v=y_valid,
                                               x_te=x_test, y_te=y_test, batch_size=1024)
                        test_acc_1 = test_score_1[1]
                        test_score_2 = train_cnn(cnn=model, x_tr=x_train, y_tr=y_train, x_v=x_test, y_v=y_test,
                                                 x_te=x_valid, y_te=y_valid, batch_size=1024)
                        test_acc_2 = test_score_2[1]
                        test_acc = (test_acc_1 + test_acc_2) / 2
                        dic = {'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'd': d}
                        res = (dic, test_acc)
                        result.append(res)
                        ctr += 1
    return result


def grid_search_parameters(best_architecture, optimizer, batch_size, dim):
    count_total = len(optimizer) * len(batch_size)
    h1 = best_architecture['h1']
    h2 = best_architecture['h2']
    h3 = best_architecture['h3']
    h4 = best_architecture['h4']
    d = best_architecture['d']
    result = []
    ctr = 1
    for o in optimizer:
        for b in batch_size:
            print('Hyperparameter setting {} out of {} ...'.format(ctr, count_total))
            model = build_cnn(n_h1=h1, n_h2=h2, n_h3=h3, n_h4=h4, n_d=d, opt=o, dim=dim)
            test_score_1 = train_cnn(cnn=model, x_tr=x_train, y_tr=y_train, x_v=x_valid, y_v=y_valid,
                                   x_te=x_test, y_te=y_test, batch_size=b)
            test_acc_1 = test_score_1[1]
            test_score_2 = train_cnn(cnn=model, x_tr=x_train, y_tr=y_train, x_v=x_test, y_v=y_test,
                                   x_te=x_valid, y_te=y_valid, batch_size=b)
            test_acc_2 = test_score_2[1]
            test_acc = (test_acc_1 + test_acc_2) / 2
            dic = {'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'd': d, 'o': o, 'b': b}
            res = (dic, test_acc)
            result.append(res)
            ctr += 1
    return result


if __name__ == '__main__':
    start = datetime.datetime.now()

    wm_dim = 64
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        ds.dataset(include_nonpattern=False, wm_dim=wm_dim)

    grid_result_arc = grid_search_architecture(hidden_1=[16, 32], hidden_2=[32, 64], hidden_3=[32, 64],
                                               hidden_4=[64, 128], dense=[128, 256], dim=wm_dim)
    grid_result_arc.sort(key=lambda x: x[1])
    best_grid_result_arc = grid_result_arc[-1]
    best_arc = best_grid_result_arc[0]

    grid_result_param = grid_search_parameters(best_architecture=best_arc, optimizer=['adam', 'nadam'],
                                               batch_size=[128, 256, 512, 1024], dim=wm_dim)
    grid_result_param.sort(key=lambda x: x[1])
    best_grid_result = grid_result_param[-1]

    print('Best settings:\n', best_grid_result[0])
    print('\nBest settings test accuracy:', best_grid_result[1])
