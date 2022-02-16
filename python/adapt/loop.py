import numpy as np
import tensorflow as tf
from adapt.ml.dl_subclass import get_test
from adapt.ml.lda import eval_lda

def test_models(x_test_cnn, x_test_mlp, x_lda, y_test, y_lda, cnn, mlp, w, c):
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

def check_labels(test_data,test_params,train_dof,key):
    # check classes trained vs tested
    test_dof = np.unique(test_params[:,2])
    test_key = np.empty(test_dof.shape)
    for dof_i in range(len(test_dof)):
        test_key[dof_i] = test_params[np.argmax(test_params[:,2] == test_dof[dof_i]),0]
    
    test_dof = test_dof[np.argsort(test_key)]
    test_key = np.sort(test_key)

    if not(np.all(np.in1d(test_dof,train_dof)) and np.all(np.in1d(train_dof,test_dof))):
        if len(test_dof) < len(train_dof):
            print('Missing classes')
            for key_i in key:
                test_params[test_params[:,2] == train_dof[int(key_i-1)],0] = key_i
        overlap = ~np.in1d(test_dof, train_dof)
        if overlap.any():
            print('Removing ' + test_dof[overlap])
            for ov_i in range(np.sum(overlap)):
                ind = test_params[:,2] == test_dof[overlap][ov_i]
                test_params[ind,:] = []
                test_data[ind,:] = []
    
    return test_data, test_params