import numpy as np
import tensorflow as tf
from adapt.ml.dl_subclass import MLP, CNN, ALI, get_train, get_test
from adapt.ml.lda import train_lda, eval_lda

def train_models(traincnn, trainmlp, x_train_lda, y_train_lda, n_dof, ep=30, mlp=None, cnn=None, print_b=False,lr=0.001, align=False):
    # Train NNs
    if mlp == None:
        mlp = MLP(n_class=n_dof)
    if cnn == None:
        cnn = CNN(n_class=n_dof)
    if align == True:
        mlp_ali = ALI()
        cnn_ali = ALI()
    else:
        mlp_ali = None
        cnn_ali = None

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    models = [mlp]
    for model in models:
        if isinstance(model,CNN):
            ds = traincnn
        else:
            ds = trainmlp
        
        if not align:
            ali = None

        train_mod = get_train()

        for epoch in range(ep):
            # Reset the metrics at the start of the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()

            for x, y, _ in ds:
                if isinstance(model,CNN):
                    train_mod(x, y, model, optimizer, train_loss, train_accuracy, align=cnn_ali)
                else:
                    train_mod(x, y, model, optimizer, train_loss, train_accuracy, align=mlp_ali)

            if print_b:
                if epoch == 0 or epoch == ep-1:
                    print(f'Epoch {epoch + 1}, ', f'Loss: {train_loss.result():.2f}, ', f'Accuracy: {train_accuracy.result() * 100:.2f} ')

    # Train LDA
    w,c, _, _, _ = train_lda(x_train_lda,y_train_lda)

    # print(align)
    if align:
        return mlp, cnn, mlp_ali, cnn_ali, w, c
    else:
        return mlp, cnn, w, c

def test_models(x_test_cnn, x_test_mlp, x_lda, y_test, y_lda, cnn, mlp, w, c, cnn_align=None, mlp_align=None):
    acc = np.empty((3,))
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    test_loss.reset_states()
    test_accuracy.reset_states()

    # test CNN
    test_mod = get_test()
    test_mod(x_test_cnn, y_test, cnn, test_loss, test_accuracy, align=cnn_align)
    acc[2] = test_accuracy.result()*100

    # test MLP
    test_loss.reset_states()
    test_accuracy.reset_states()

    test_mod = get_test()
    test_mod(x_test_mlp, y_test, mlp, test_loss, test_accuracy, align=mlp_align)
    acc[1] = test_accuracy.result()*100

    # test LDA
    acc[0] = eval_lda(w, c, x_lda, y_lda) * 100

    return acc

def check_labels(test_data,test_params,train_dof,key):
    # check classes trained vs tested
    test_dof = np.unique(test_params[:,2])
    test_key = np.empty(test_dof.shape)
    for dof_i in range(len(test_dof)):
        test_key[dof_i] = test_params[np.argmax(test_params[:,2] == test_dof[dof_i]),0]

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