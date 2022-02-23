import numpy as np
import tensorflow as tf
from adapt.ml.dl_subclass import MLP, CNN, ALI, get_train, get_test, EWC, get_train_ewc, get_fish
from adapt.ml.lda import train_lda, eval_lda
import matplotlib.pyplot as plt
from IPython import display

# classification accuracy plotting
def plot_test_acc(plot_handles):
    plt.legend(handles=plot_handles, loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.ylim(0,1)
    # display.display(plt.gcf())
    # display.clear_output(wait=True)
    
# train/compare vanilla sgd and ewc
def train_task(model, num_iter, disp_freq, x_train, y_train, x_test, y_test, lams=[0]):
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        model.restore() # reassign optimal weights from previous training session
        test_accs = []
        for task in range(len(x_test)):
            test_accs.append(np.zeros(int(num_iter/disp_freq)))
        train_ewc = get_train_ewc()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0],reshuffle_each_iteration=True).batch(32)
        # train on current task
        train_last = -9999
        for iter in range(num_iter):
            for x_in, y_in in ds:
                train_ewc(x_in, y_in, model, optimizer, train_loss, train_accuracy, lam=lams[l])

            if iter % disp_freq == 0:
                for task in range(len(x_test)):
                    test_accs[task][int(iter/disp_freq)] = model.acc(x=x_test[task], y = y_test[task])

            train_diff = train_loss.result() - train_last
            train_last = train_loss.result()
            print(train_diff)
            if train_diff < 0 and train_diff > -1e-3:
                break

        plt.subplot(1, len(lams), l+1)
        colors = ['r', 'b', 'g']
        plots = []
        for task in range(len(x_test)):
            c = chr(ord('A') + task)
            if disp_freq > iter:
                disp_freq = 1
            plot_h, = plt.plot(range(0,iter+1,disp_freq), test_accs[task][:iter+1], colors[task], label="task " + c)
            plots.append(plot_h)
            print(f'Acc: {test_accs[task][iter]:.4f}')
        plot_test_acc(plots)
        if l == 0: 
            plt.title("vanilla sgd")
        else:
            plt.title("ewc")
        plt.gcf().set_size_inches(len(lams)*5, 3.5)

def train_models(traincnn, trainmlp, x_train_lda, y_train_lda, n_dof, ep=30, mlp=None, cnn=None, print_b=False,lr=0.001, align=False):
    # Train NNs
    if mlp == None:
        # mlp = MLP(n_class=n_dof)
        mlp = EWC(n_class=n_dof)
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
    if isinstance(x_train_lda,np.ndarray):
        w,c, _, _, _ = train_lda(x_train_lda,y_train_lda)
    else:
        w=0
        c=0

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