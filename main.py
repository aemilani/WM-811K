import os
import datetime
import numpy as np
import dataset as ds
import models as md
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import setup_logger


date, time = str(datetime.datetime.now()).split()

dirr = 'results/{}/{}/'.format(date, time.replace(':', '-'))
os.makedirs(dirr)
cp_path = os.path.join(dirr, 'checkpoints')
log_path = os.path.join(dirr, 'logs')
os.mkdir(cp_path)
os.mkdir(log_path)

logger = setup_logger('main', log_path)

wm_dim = 64
np.random.seed(13)

# Binary classification

for test_ratio, valid_ratio in zip([0.2, 0.2, 0.3, 0.3], [0.2, 0.3, 0.2, 0.3]):
    start_binary = datetime.datetime.now()
    logger.info('Loading dataset_binary. test_ratio={}, valid_ratio={}'.format(test_ratio, valid_ratio))
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        ds.dataset_binary(wm_dim=wm_dim, test_ratio=test_ratio, valid_ratio=valid_ratio)
    duration_data_binary = datetime.datetime.now() - start_binary
    logger.info('dataset_binary loaded successfully. Duration: {}'.format(duration_data_binary))

    cnn = md.cnn_binary(wm_dim=wm_dim)

    ea = EarlyStopping(patience=5)
    cp = ModelCheckpoint(os.path.join(cp_path, 'cnn_binary_{}_{}.h5'.format(test_ratio, valid_ratio)),
                         save_best_only=True)
    cb = [ea, cp]

    logger.info('Training starting ...')
    history = cnn.fit(x_train,
                      y_train, batch_size=512, epochs=30,
                      validation_data=(x_valid, y_valid),
                      callbacks=cb)
    duration_train_binary = datetime.datetime.now() - start_binary - duration_data_binary
    logger.info('Training ended. Duration: {}'.format(duration_train_binary))

    cnn = load_model(os.path.join(cp_path, 'cnn_binary_{}_{}.h5'.format(test_ratio, valid_ratio)))

    plt.figure()
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Categorical Cross-Entropy - Binary Classification - Test Ratio={}, Valid Ratio={}'.format(
        test_ratio, valid_ratio))
    plt.savefig(os.path.join(dirr, 'loss_binary_{}_{}.png'.format(test_ratio, valid_ratio)))
    plt.show()

    plt.figure()
    plt.plot(history.history['categorical_accuracy'], label='Training accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy - Binary Classification - Test Ratio={}, Valid Ratio={}'.format(
        test_ratio, valid_ratio))
    plt.savefig(os.path.join(dirr, 'accuracy_binary_{}_{}.png'.format(test_ratio, valid_ratio)))
    plt.show()

    logger.info('Evaluation starting ...')

    score = cnn.evaluate(x_test, y_test, verbose=0)
    logger.info('Test loss: {}'.format(score[0]))
    logger.info('Test accuracy: {}'.format(score[1]))

    predictions = cnn.predict(x_test)

    cm_binary = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
    cm_norm_binary = confusion_matrix(np.argmax(y_test, axis=1),
                                      np.argmax(predictions, axis=1), normalize='true')
    cm_norm_binary = np.around(cm_norm_binary, 3)

    np.savetxt(os.path.join(dirr, 'cm_binary_{}_{}.csv'.format(test_ratio, valid_ratio)), cm_binary, delimiter=',')
    np.savetxt(os.path.join(dirr, 'cm_norm_binary_{}_{}.csv'.format(
        test_ratio, valid_ratio)), cm_norm_binary, delimiter=',')

    duration_eval_binary = datetime.datetime.now() - start_binary - duration_data_binary - duration_train_binary

    logger.info('Confusion matrix:\n{}'.format(cm_binary))
    logger.info('Normalized confusion matrix:\n{}'.format(cm_norm_binary))
    logger.info('Evaluation ended. Duration: {}'.format(duration_eval_binary))

    duration_binary = datetime.datetime.now() - start_binary
    logger.info('Binary classification run time: {}'.format(duration_binary))


# Multi-class classification

for test_ratio, valid_ratio in zip([0.2, 0.2, 0.3, 0.3], [0.2, 0.3, 0.2, 0.3]):
    start_multi = datetime.datetime.now()
    logger.info('Loading dataset_multi. test_ratio={}, valid_ratio={}'.format(test_ratio, valid_ratio))
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        ds.dataset_multi(wm_dim=wm_dim, test_ratio=test_ratio, valid_ratio=valid_ratio)
    duration_data_multi = datetime.datetime.now() - start_multi
    logger.info('dataset_multi loaded successfully. Duration: {}'.format(duration_data_multi))

    cnn = md.cnn_multi(wm_dim=wm_dim)

    ea = EarlyStopping(patience=5)
    cp = ModelCheckpoint(os.path.join(cp_path, 'cnn_multi_{}_{}.h5'.format(test_ratio, valid_ratio)),
                         save_best_only=True)
    cb = [ea, cp]

    logger.info('Training starting ...')
    history = cnn.fit(x_train,
                      y_train, batch_size=512, epochs=30,
                      validation_data=(x_valid, y_valid),
                      callbacks=cb)
    duration_train_multi = datetime.datetime.now() - start_multi - duration_data_multi
    logger.info('Training ended. Duration: {}'.format(duration_train_multi))

    cnn = load_model(os.path.join(cp_path, 'cnn_multi_{}_{}.h5'.format(test_ratio, valid_ratio)))

    plt.figure()
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Categorical Cross-Entropy - Multi-Class Classification - Test Ratio={}, Valid Ratio={}'.format(
        test_ratio, valid_ratio))
    plt.savefig(os.path.join(dirr, 'loss_multi_{}_{}.png'.format(test_ratio, valid_ratio)))
    plt.show()

    plt.figure()
    plt.plot(history.history['categorical_accuracy'], label='Training accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy - Multi-Class Classification - Test Ratio={}, Valid Ratio={}'.format(
        test_ratio, valid_ratio))
    plt.savefig(os.path.join(dirr, 'accuracy_multi_{}_{}.png'.format(test_ratio, valid_ratio)))
    plt.show()

    logger.info('Evaluation starting ...')

    score = cnn.evaluate(x_test, y_test, verbose=0)
    logger.info('Test loss: {}'.format(score[0]))
    logger.info('Test accuracy: {}'.format(score[1]))

    predictions = cnn.predict(x_test)

    cm_multi = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
    cm_norm_multi = confusion_matrix(np.argmax(y_test, axis=1),
                                     np.argmax(predictions, axis=1), normalize='true')
    cm_norm_multi = np.around(cm_norm_multi, 3)

    np.savetxt(os.path.join(dirr, 'cm_multi_{}_{}.csv'.format(test_ratio, valid_ratio)), cm_multi, delimiter=',')
    np.savetxt(os.path.join(dirr, 'cm_norm_multi_{}_{}.csv'.format(
        test_ratio, valid_ratio)), cm_norm_multi, delimiter=',')

    duration_eval_multi = datetime.datetime.now() - start_multi - duration_data_multi - duration_train_multi

    logger.info('Confusion matrix:\n{}'.format(cm_multi))
    logger.info('Normalized confusion matrix:\n{}'.format(cm_norm_multi))
    logger.info('Evaluation ended. Duration: {}'.format(duration_eval_multi))

    duration_multi = datetime.datetime.now() - start_multi
    logger.info('Multi-class classification run time: {}'.format(duration_multi))
