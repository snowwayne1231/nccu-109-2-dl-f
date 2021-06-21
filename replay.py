import numpy as np
import tensorflow as tf
import json
import operator
from tensorflow import keras
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


if __name__ == '__main__':
    model = tf.keras.models.load_model('saved_model.h5')
    model.summary()
    
    _val_x = np.load('_val_x.npy')
    _val_y = np.load('_val_y.npy')
    # _train_x = np.load('_train_x.npy')
    # _train_y = np.load('_train_y.npy')

    predictions = model.predict(_val_x)
    # predictions = model.predict(_train_x)

    total = len(predictions)
    cm_map = np.ndarray((5,5), dtype=int)
    cm_map.fill(0)
    right = 0
    for pidx in range(total):
        pp = predictions[pidx]
        y = _val_y[pidx]
        # y = _train_y[pidx]
        p_ans = np.argmax(pp)
        ans = np.argmax(y)
        cm_map[ans][p_ans] += 1
        if p_ans == ans:
            right += 1
    
    print("accuracy: {:5.2f}%".format(100 * right / total))
    plot_confusion_matrix(cm_map, ['none','big down','down','up','huge up'])


