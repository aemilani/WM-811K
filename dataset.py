import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from utils import resize_wafer_map, one_hot_img, add_noise


def dataset(include_nonpattern=False):
    df = pd.read_pickle("data/LSWMD.pkl")
    df = df.drop(['trianTestLabel'], axis=1)
    df = df.drop(['waferIndex'], axis=1)

    df['failureNum'] = df.failureType
    mapping = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4, 'Random': 5, 'Scratch': 6,
               'Near-full': 7, 'none': 8}
    df = df.replace({'failureNum': mapping})

    if include_nonpattern:
        failure_types = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'none']
        df = df[(df['failureNum'] >= 0) & (df['failureNum'] <= 8)]
    else:
        failure_types = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
        df = df[(df['failureNum'] >= 0) & (df['failureNum'] <= 7)]

    target_dim = 64
    df['waferMapResized'] = df.waferMap.apply(lambda w: resize_wafer_map(w, target_size=(target_dim, target_dim))).\
        apply(one_hot_img)

    for label in range(len(failure_types)):
        globals()['df_{}'.format(label)] = df[(df['failureNum'] == label)].sample(frac=1)

    test_ratio, valid_ratio = 0.2, 0.2
    for label in range(len(failure_types)):
        length = len(globals()['df_{}'.format(label)])
        idx_test = int(length * test_ratio)
        idx_valid = idx_test + int(length * valid_ratio)
        globals()['df_{}_test'.format(label)] = globals()['df_{}'.format(label)][:idx_test].reset_index()
        globals()['df_{}_valid'.format(label)] = globals()['df_{}'.format(label)][idx_test:idx_valid].reset_index()
        globals()['df_{}_train'.format(label)] = globals()['df_{}'.format(label)][idx_valid:].reset_index()
        del globals()['df_{}'.format(label)]

    max_class_len_train = np.max([len(globals()['df_{}_train'.format(i)]) for i in range(len(failure_types))])

    x_test, y_test = [], []
    for label in range(len(failure_types)):
        for wm in globals()['df_{}_test'.format(label)].waferMapResized:
            x_test.append(np.expand_dims(wm, axis=0))
            y_test.append(label)

    x_valid, y_valid = [], []
    for label in range(len(failure_types)):
        for wm in globals()['df_{}_valid'.format(label)].waferMapResized:
            x_valid.append(np.expand_dims(wm, axis=0))
            y_valid.append(label)

    x_train, y_train = [], []
    for label in range(len(failure_types)):
        length = len(globals()['df_{}_train'.format(label)])
        n_new = max_class_len_train - length
        for i in range(n_new):
            x = globals()['df_{}_train'.format(label)].waferMapResized.loc[np.random.choice(range(length))]
            if np.random.random() < 0.5:
                new_x = tf.image.flip_left_right(x).numpy()
            else:
                new_x = tf.image.rot90(x).numpy()
            new_x = add_noise(new_x)
            x_train.append(np.expand_dims(new_x, axis=0))
            y_train.append(label)
        for wm in globals()['df_{}_train'.format(label)].waferMapResized:
            x_train.append(np.expand_dims(wm, axis=0))
            y_train.append(label)

    x_test_arr = np.concatenate(x_test, axis=0)
    y_test_arr = np.array(y_test)
    x_valid_arr = np.concatenate(x_valid, axis=0)
    y_valid_arr = np.array(y_valid)
    x_train_arr = np.concatenate(x_train, axis=0)
    y_train_arr = np.array(y_train)
    del x_test, y_test, x_valid, y_valid, x_train, y_train

    y_test_arr = to_categorical(y_test_arr)
    y_valid_arr = to_categorical(y_valid_arr)
    y_train_arr = to_categorical(y_train_arr)

    return x_train_arr, y_train_arr, x_valid_arr, y_valid_arr, x_test_arr, y_test_arr
