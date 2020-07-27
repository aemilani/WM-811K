import os
import logging
import numpy as np
import tensorflow as tf


def setup_logger(logger_name, log_path):
    log_file_path = os.path.join(log_path, '{}.log'.format(logger_name.lower()))
    file_format = logging.Formatter('[%(asctime)s] %(levelname)-8s %(message)s',
                                    datefmt='%Y/%m/%d %H:%M:%S')
    console_format = logging.Formatter('%(levelname)-8s %(message)s')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(file_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(console_format)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


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
