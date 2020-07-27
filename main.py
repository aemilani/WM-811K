import datetime
import numpy as np
import dataset as ds
import models as md
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix


start = datetime.datetime.now()

wm_dim = 64
x_train, y_train, x_valid, y_valid, x_test, y_test = ds.dataset(wm_dim=wm_dim)

duration_data = datetime.datetime.now() - start
print('Dataset loaded successfully. Duration: {}'.format(duration_data))

cnn = md.cnn(wm_dim=wm_dim)

ea = EarlyStopping(patience=3)
cp = ModelCheckpoint('checkpoints/cnn.h5', save_best_only=True)
cb = [ea, cp]

history = cnn.fit(np.expand_dims(x_train[:, :, :, 2], axis=-1),
                  y_train, batch_size=1024, epochs=30,
                  validation_data=(np.expand_dims(x_valid[:, :, :, 2], axis=-1), y_valid),
                  callbacks=cb)

plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Categorical Cross-Entropy')
plt.show()

plt.figure()
plt.plot(history.history['categorical_accuracy'], label='Training accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()

score = cnn.evaluate(np.expand_dims(x_test[:, :, :, 2], axis=-1), y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = cnn.predict(np.expand_dims(x_test[:, :, :, 2], axis=-1))

cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
cm_norm = confusion_matrix(np.argmax(y_test, axis=1),
                           np.argmax(predictions, axis=1), normalize='true')
cm_norm = np.around(cm_norm, 2)
print('Confusion matrix:\n', cm_norm)

duration_train = datetime.datetime.now() - start - duration_data
print('Training took {}.'.format(duration_train))
