import numpy as np
import tensorflow as tf


def resize_wafer_map(wm, target_size=(64, 64)):
    return tf.image.resize(np.expand_dims(wm, axis=-1), target_size, method='nearest').numpy()


def one_hot_img(wm):
    h, w = wm.shape[0], wm.shape[1]
    one_hot = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            one_hot[i, j, int(wm[i, j])] = 1
    return one_hot


def add_noise(wm):
    h, w = wm.shape[0], wm.shape[1]
    n_change = int(0.01 * h * w)
    changed = 0
    noised = wm.copy()
    while changed < n_change:
        i, j = np.random.randint(0, h), np.random.randint(0, w)
        if wm[i, j, 0] == 0:  # If the point is on the wafer
            noised[i, j, 1] = int(not noised[i, j, 1])
            noised[i, j, 2] = int(not noised[i, j, 2])
            changed += 1
    return noised
