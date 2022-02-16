import numpy as np
import tensorflow as tf
from ml.dl_subclass import get_test
from ml.lda import eval_lda

def test_loop(x_test_cnn, x_test_mlp, x_lda, y_test, y_lda, cnn, mlp, w, c):
    acc = np.empty((3,))
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    test_loss.reset_states()
    test_accuracy.reset_states()

    # test CNN
    test_mod = get_test()
    test_mod(x_test_cnn, y_test, cnn, test_loss, test_accuracy)
    acc[0] = test_accuracy.result()*100

    # test MLP
    test_loss.reset_states()
    test_accuracy.reset_states()

    test_mod = get_test()
    test_mod(x_test_mlp, y_test, mlp, test_loss, test_accuracy)
    acc[1] = test_accuracy.result()*100

    # test LDA
    acc[2] = eval_lda(w, c, x_lda, y_lda) * 100

    return acc