import tensorflow as tf
import keras
import numpy as np
import os
import keras.backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def loss_func(targets, logits):
    '''
    Parameters:
        targets: The targets
            Shape: (batch_size, max_pr_len)
        logits: The logits
            Shape: (batch_size, max_pr_len, vocab_size)
    '''
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Mask padding values, they do not have to compute for loss
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    
    # Calculate the loss value
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss

def accuracy_fn(y_true, y_pred):
    '''
    Parameters:
        y_true: The true labels
            Shape: (batch_size, max_len)
        y_pred: The predicted labels
            Shape: (batch_size, max_len, vocab_size)
    '''
    pred_values = K.cast(K.argmax(y_pred, axis=-1), dtype='int32')
    correct = K.cast(K.equal(y_true, pred_values), dtype='float32')

    # 1 is padding, don't include those
    # mask = K.cast(K.greater(y_true, 0), dtype='float32')
    mask = K.cast(y_true != 1, dtype='float32')
    n_correct = K.sum(mask * correct)
    n_total = K.sum(mask)
  
    return n_correct / n_total

if __name__ == '__main__':
    # Generate random data
    y_true = np.random.randint(0, 100, (3, 10))
    y_pred = np.random.normal(0, 1, (3, 10, 100))

    print(loss_func(y_true, y_pred))
    print(accuracy_fn(y_true, y_pred))