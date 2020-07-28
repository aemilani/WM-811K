import os
import datetime
import numpy as np
import dataset as ds
import models as md
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from utils import setup_logger


start = datetime.datetime.now()
date, time = str(start).split()


dirr = 'results/{}/{}/'.format(date, time.replace(':', '-'))
os.makedirs(dirr)
cp_path = os.path.join(dirr, 'checkpoints')
log_path = os.path.join(dirr, 'logs')
os.mkdir(cp_path)
os.mkdir(log_path)

logger = setup_logger('main', log_path)

wm_dim = 128

logger.info('Loading the dataset ...')
x_train, y_train, x_valid, y_valid, x_test, y_test = \
    ds.dataset(include_nonpattern=False, wm_dim=wm_dim)
duration_data = datetime.datetime.now() - start
logger.info('Dataset loaded successfully. Duration: {}'.format(duration_data))

cnn = md.cnn(wm_dim=wm_dim)

ea = EarlyStopping(patience=5)
cp = ModelCheckpoint(os.path.join(cp_path, 'cnn.h5'), save_best_only=True)
cb = [ea, cp]

logger.info('Training starting ...')
history = cnn.fit(x_train,
                  y_train, batch_size=1024, epochs=30,
                  validation_data=(x_valid, y_valid),
                  callbacks=cb)
duration_train = datetime.datetime.now() - start - duration_data
logger.info('Training ended. Duration: {}'.format(duration_train))

plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Categorical Cross-Entropy')
plt.savefig(os.path.join(dirr, 'loss.png'))
plt.show()

plt.figure()
plt.plot(history.history['categorical_accuracy'], label='Training accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')
plt.savefig(os.path.join(dirr, 'accuracy.png'))
plt.show()

logger.info('Evaluation starting ...')

score = cnn.evaluate(x_test, y_test, verbose=0)
logger.info('Test loss: {}'.format(score[0]))
logger.info('Test accuracy: {}'.format(score[1]))

predictions = cnn.predict(x_test)

cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
cm_norm = confusion_matrix(np.argmax(y_test, axis=1),
                           np.argmax(predictions, axis=1), normalize='true')
cm_norm = np.around(cm_norm, 3)

np.savetxt(os.path.join(dirr, 'cm.csv'), cm, delimiter=',')
np.savetxt(os.path.join(dirr, 'cm_norm.csv'), cm_norm, delimiter=',')

duration_eval = datetime.datetime.now() - start - duration_data - duration_train

logger.info('Confusion matrix:\n{}'.format(cm))
logger.info('Normalized confusion matrix:\n{}'.format(cm_norm))
logger.info('Evaluation ended. Duration: {}'.format(duration_eval))

duration_total = datetime.datetime.now() - start
logger.info('Total run time: {}'.format(duration_total))
