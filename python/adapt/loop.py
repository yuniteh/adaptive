import numpy as np
import tensorflow as tf
from adapt.ml.dl_subclass import MLP, CNN, ALI, get_train, get_test, EWC
from adapt.ml.lda import train_lda, eval_lda
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
    
# train/compare vanilla sgd and ewc
def train_task(model, num_iter, disp_freq, x_train, y_train, x_test, y_test, lams=[0]):
    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0],reshuffle_each_iteration=True).batch(100)
    fig, ax = plt.subplots(len(lams),3,squeeze=False,figsize=(18,len(lams)*3.5))
    loss = np.zeros((int(num_iter/disp_freq)+1,len(lams)))
    f_loss = np.zeros((int(num_iter/disp_freq)+1,len(lams)))
    lams_all = np.zeros((int(num_iter/disp_freq)+1,len(lams)))
    for l in range(len(lams)):
        lam_in = np.abs(lams[l])
        # lams[l] sets weight on old task(s)
        model.restore() # reassign optimal weights from previous training session

        test_accs = []
        train_accs = np.zeros(int(num_iter/disp_freq)+1)
        for task in range(len(x_test)):
            test_accs.append(np.zeros(int(num_iter/disp_freq)+1))

        if lams[l] == 0:
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)
        
        # train functions
        train_ewc = get_train()
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        fish_loss = tf.keras.metrics.Mean(name='fish_loss')
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

        # validation functions
        val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
        
        # initial loss and accuracies
        train_accs[0] = train_accuracy(model(x_train),y_train)
        print(f'Initial train acc: {train_accs[0]:.4f},', end=' ')
        for task in range(len(x_test)):
            val_accuracy.reset_states()
            test_accs[task][0] = val_accuracy(model(x_test[task]),y_test[task])

            if task != len(x_test)-1:
                end_p = ', '
            else:
                end_p = '\n'
            print(f'val acc {task:d}: {test_accs[task][0]:.4f}', end = end_p)

        train_last = 9999

        # train on current task
        for iter in range(num_iter):
            train_loss.reset_states()
            fish_loss.reset_states()
            train_accuracy.reset_states()
            val_accuracy.reset_states()

            # x_train, y_train = shuffle(x_train, y_train)
            # x_in = x_train[:100,...]
            # y_in = y_train[:100,...]

            # if iter < 10 or (iter < 30 and iter > 20) or (iter < 50 and iter > 40):
            #     lam_in = 0
            # else:
            #     lam_in = lams[l]
            # if iter == 0:
            #     lam_in = 0

            for x_in, y_in in ds:
                train_ewc(x_in, y_in, model, optimizer, train_loss, fish_loss, train_accuracy, lam=lam_in)
                if f_loss[0,l] == 0:
                    loss[0,l] = train_loss.result()
                    f_loss[0,l] = fish_loss.result()
                
            if lams[l] < 0 and f_loss[iter,l] != 0:
                lam_in = np.float32(loss[iter,l]/f_loss[iter,l])
            
            lams_all[int(iter/disp_freq)+1,l] = lam_in

            if iter % disp_freq == 0:
                loss[int(iter/disp_freq)+1,l] = train_loss.result()
                f_loss[int(iter/disp_freq)+1,l] = fish_loss.result()
                train_accuracy(model(x_train),y_train)
                train_accs[int(iter/disp_freq)+1] = train_accuracy.result()
                for task in range(len(x_test)):
                    val_accuracy.reset_states()
                    test_accs[task][int(iter/disp_freq)+1] = val_accuracy(model(x_test[task]),y_test[task])

            # early stopping criteria
            train_diff = train_last - train_loss.result()
            train_last = train_loss.result()
            if train_diff > 0 and train_diff < 1e-4:
                break
        
        print(f'Final train acc: {train_accs[iter+1]:.4f}, ', end=' '),
        for task in range(len(x_test)):       
            if task != len(x_test)-1:
                end_p = ', '
            else:
                end_p = '\n'
            print(f'val acc {task:d}: {test_accs[task][iter+1]:.4f}', end=end_p)
        
        # plot results
        colors = ['r', 'b', 'g']
        if disp_freq > iter:
            disp_freq = 1

        ax[l,0].plot(range(0,iter+2,disp_freq), loss[:iter+2,l], 'r-', label="class loss")
        if np.sum(f_loss[:,l]) > 0:
            ax[l,0].plot(range(0,iter+2,disp_freq), f_loss[:iter+2,l], 'b-', label="fish loss")
        ax[l,0].legend(loc="center right")
        ax[l,0].set_ylabel("Loss")

        ax[l,1].plot(range(0,iter+2,disp_freq), train_accs[:iter+2], 'b-')
        ax[l,1].set_ylabel("Train Accuracy")
        ax[l,1].set_ylim((0,1.1))

        for task in range(len(x_test)):
            c = chr(ord('A') + task)
            ax[l,2].plot(range(0,iter+2,disp_freq), test_accs[task][:iter+2], colors[task], label="task " + c)
        ax[l,2].legend(loc="center right")
        ax[l,2].set_ylabel("Valid Accuracy")
        ax[l,2].set_ylim((0,1.1))

        if l == 0:
            ax[l,1].set_title('Vanilla MLP')
        else:
            ax[l,1].set_title('EWC (λ: ' + str(lams[l]) + ')')

        for i in range(3):
            if l == len(lams)-1:
                ax[l,i].set_xlabel("Iterations")
            # else:
            #     ax[l,i].set_xlabel(().set_visible(False)
    
    return loss, f_loss


def train_models(traincnn, trainmlp, x_train_lda, y_train_lda, n_dof, ep=30, mlp=None, cnn=None, print_b=False,lr=0.001, align=False):
    # Train NNs
    if mlp == None:
        mlp = MLP(n_class=n_dof)
        # mlp = EWC(n_class=n_dof)
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
    models = [mlp,cnn]
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

    # test_loss.reset_states()
    # test_accuracy.reset_states()
    # test_mod = get_test()
    # test_mod(x_test_mlp, y_test, w, test_loss, test_accuracy, align=mlp_align)
    # acc[0] = test_accuracy.result()*100

    #test LDA
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