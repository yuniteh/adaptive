import numpy as np
import tensorflow as tf
from adapt.ml.dl_subclass import CNN, VCNN, get_train, get_test
from adapt.ml.lda import train_lda, eval_lda
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from AdaBound2 import AdaBound as AdaBoundOptimizer
import copy as cp
import time
from tensorflow.keras import mixed_precision

    
# train/compare vanilla sgd and ewc
def train_task(model, num_iter, disp_freq, x_train, y_train, x_test=[], y_test=None, lams=[0], plot_loss=True, bat=32, clda=None, cnnlda=False):
    # bat = 128
    bat=32
    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0],reshuffle_each_iteration=True).batch(bat)

    if plot_loss:
        _, ax = plt.subplots(len(lams),3,squeeze=False,figsize=(18,len(lams)*3.5))
    
    loss = np.zeros((int(num_iter/disp_freq)+1,len(lams)))
    f_loss = np.zeros((int(num_iter/disp_freq)+1,len(lams)))
    lams_all = np.zeros((int(num_iter/disp_freq)+1,len(lams)))
    
    # validation functions
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
    val_mod = get_test(model,val_accuracy)

    for l in range(len(lams)):
        lam_in = np.abs(lams[l])
        if lams[l] > 0:
            lam_array = np.arange(1,lams[l],lams[l]//5)
        # lams[l] sets weight on old task(s)
        model.restore() # reassign optimal weights from previous training session

        test_accs = []
        for task in range(len(x_test)):
            test_accs.append(np.zeros(int(num_iter/disp_freq)+1))
        
        if lams[l] == 0:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            # optimizer = tf.keras.optimizers.SGD(learning_rate=0.000001)#,clipvalue=.5)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
            # optimizer = tf.keras.optimizers.SGD(learning_rate=0.000001)#,clipvalue=.5)
        
        # train functions
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        fish_loss = tf.keras.metrics.Mean(name='fish_loss')
        train_ewc = get_train()
        # initial loss and accuracies
        print(f'Initial', end=' ')
        for task in range(len(x_test)):
            val_accuracy.reset_states()
            val_mod(tf.convert_to_tensor(x_test[task]), tf.convert_to_tensor(y_test[task]))
            test_accs[task][0] = val_accuracy.result()

            if task < len(x_test)-1:
                end_p = ', '
            else:
                end_p = '\n'
            print(f'val acc {task:d}: {test_accs[task][0]:.4f}', end = end_p)

        train_last = 9999
        fish_last = 9999

        start_time = time.time()
        # train on current task
        for iter in range(num_iter):
            train_loss.reset_states()
            fish_loss.reset_states()
            val_accuracy.reset_states()

            # if iter == 0 and lams[l] > 0:
            #     lam_in = 1
            if lams[l] > 0:
                if iter < 5:
                    # lam_in = lam_array[iter]
                    lam_in = lams[l]
                else:
                    lam_in = lams[l]

            for x_in, y_in in ds:
                train_ewc(x_in, y_in, model, optimizer, train_loss, fish_loss, lam=lam_in, clda=clda, trainable=False)
                
                if f_loss[0,l] == 0:
                    loss[0,l] = train_loss.result()
                    f_loss[0,l] = fish_loss.result()
                    if lams[l] > 0:
                        print('loss:' + str(train_loss.result().numpy()) + ', fish: ' + str(fish_loss.result().numpy()) + ', lam: ' + str(lam_in))
            
            ratio = 0
            ## weight cycling
            # if lams[l] != 0:
            #     ratio = (train_loss.result()/fish_loss.result()).numpy()
            #     lam_in = lams[l]
            #     optimizer.learning_rate = 0.000001
            
            lams_all[int(iter/disp_freq)+1,l] = lam_in

            if iter % disp_freq == 0:
                loss[int(iter/disp_freq)+1,l] = train_loss.result()
                f_loss[int(iter/disp_freq)+1,l] = fish_loss.result()
                for task in range(len(x_test)):
                    val_accuracy.reset_states()
                    val_mod(tf.convert_to_tensor(x_test[task]), tf.convert_to_tensor(y_test[task]))
                    test_accs[task][int(iter/disp_freq)+1] = val_accuracy.result()

            # early stopping criteria
            train_diff = train_last - train_loss.result()
            train_last = cp.deepcopy(train_loss.result().numpy())
            fish_diff = fish_last - fish_loss.result()
            fish_last = cp.deepcopy(fish_loss.result().numpy())
            if train_diff > 0 and train_diff < 1e-4:
                print('early stop')
                break
            
            if lams[l] > 0:
                if np.abs(train_diff) < 1e-3 and fish_diff < 0:
                    print('early stop')
                    break
            else:
                if np.abs(train_diff) < 25e-3:
                    print('early stop')
                    break
            #     # if np.abs(fish_diff) < 1e-3 and np.abs(train_diff) < 1e-3 and train_loss.result() < 1:
            #     #     # print(str(fish_loss.result().numpy()))
            #     #     # print(str(train_loss.result().numpy()))
            #     #     print('early stop fish')
            #         break

            if lams[l] > 0:
                print('loss:' + str(train_loss.result().numpy()) + ', fish: ' + str(fish_loss.result().numpy()) + ', lam: ' + str(lam_in) + ', rat: ' + str(ratio))
        elapsed = time.time()-start_time
        print('time: ' + str(elapsed))
        if cnnlda:
            x_train1 = x_train[:x_train.shape[0]//2,...]
            x_train2 = x_train[x_train.shape[0]//2:,...]
            x_lda = np.vstack((model.enc(x_train1).numpy(),model.enc(x_train2).numpy()))
            y_lda = np.vstack((y_train[:x_train.shape[0]//2,...], y_train[x_train.shape[0]//2:,...]))
            w, c, _, _, _, _, _ = train_lda(x_lda,np.argmax(y_lda,axis=1)[...,np.newaxis])
        else:
            w = 0
            c = 0

        print(f'Final', end=' '),
        for task in range(len(x_test)):       
            if task < len(x_test)-1:
                end_p = ', '
            else:
                end_p = '\n'
            print(f'val acc {task:d}: {test_accs[task][iter+1]:.4f}', end=end_p)
        
        # plot results
        colors = ['r', 'b', 'g']
        if disp_freq > iter:
            disp_freq = 1
        
        if plot_loss:
            ax[l,0].plot(range(0,iter+2,disp_freq), loss[:iter+2,l], 'r-', label="class loss")
            if np.sum(f_loss[:,l]) > 0:
                ax[l,1].plot(range(0,iter+2,disp_freq), f_loss[:iter+2,l], 'b-', label="fish loss")
            ax[l,1].legend(loc="center right")
            ax[l,1].set_ylabel("Loss")

            for task in range(len(x_test)):
                col = chr(ord('A') + task)
                ax[l,2].plot(range(0,iter+2,disp_freq), test_accs[task][:iter+2], colors[task], label="task " + col)
            ax[l,2].legend(loc="center right")
            ax[l,2].set_ylabel("Valid Accuracy")
            ax[l,2].set_ylim((0,1.1))

            if lams[l] == 0:
                ax[l,0].set_title('Vanilla MLP')
            else:
                ax[l,0].set_title('EWC (λ: ' + str(lams[l]) + ')')

            for i in range(3):
                if l == len(lams)-1:
                    ax[l,i].set_xlabel("Iterations")
                ax[l,i].set_xlim([0,iter+2])
    
        tf.keras.backend.clear_session()
    plt.show()
    
    return w, c, elapsed

def train_models(traincnn=None, y_train=None, x_train_lda=None, y_train_lda=None, n_dof=7, ep=30, mod=None, cnnlda=False, adapt=False, print_b=False, lr=0.001, bat=128, dec=True, bn_training=True, bn_trainable = True, prog_train=True):
    # Train NNs
    out = []
    for model in mod:
        if model == 'lda':
            w,c, _, _, _, _, _ = train_lda(x_train_lda,y_train_lda)
            out.extend([w,c])
        else:
            w_c = None
            ds = tf.data.Dataset.from_tensor_slices((traincnn, y_train, y_train)).shuffle(traincnn.shape[0],reshuffle_each_iteration=True).batch(bat)
            if isinstance(model,CNN):# adapting CNN
                trainable = False
            elif model == 'cnn': # calibrating CNN
                model = CNN(n_class=n_dof, adapt=adapt)
                trainable = True
                    
            elif isinstance(model,list): # calibrating CNN-LDA
                w_c = model[1:3]
                model = model[0]

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            # optimizer = tf.keras.optimizers.SGD(learning_rate=.0001)
            train_loss = tf.keras.metrics.Mean(name='train_loss')
            sec_loss = tf.keras.metrics.Mean(name='sec_loss')
            kl_loss = tf.keras.metrics.Mean(name='kl_loss')
            train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
            
            train_mod = get_train()

            print('training nn')
            start_time = time.time()
            
            for epoch in range(ep):
                # Reset the metrics at the start of the next epoch
                train_loss.reset_states()
                train_accuracy.reset_states()
                sec_loss.reset_states()
                kl_loss.reset_states()
                
                if ep > 20:
                    if epoch > 15:
                        lam_in = [10,0]
                    else:
                        lam_in = [10,0]
                else:
                    lam_in = [100,10]

                for x, y, _ in ds:
                    train_mod(x, y, model, optimizer, train_loss, train_accuracy=train_accuracy, clda=w_c, bn_training=bn_training, trainable=bn_trainable, adapt=adapt, prog_train=prog_train)

                if print_b:
                    # if epoch == 0 or epoch == ep-1:
                    print(f'Epoch {epoch + 1}, ', f'Loss: {train_loss.result():.2f}, ',f'Second Loss: {sec_loss.result():.2f}, ',f'KL Loss: {kl_loss.result():.2f}, ', f'Accuracy: {train_accuracy.result() * 100:.2f} ')
            
            elapsed = time.time() - start_time
            print('time: ' + str(elapsed))
            out.extend([model,elapsed])

            

            if isinstance(model,CNN):
                if cnnlda:
                    w_c,c_c, _, _, _, _, _ = train_lda(model.enc(traincnn),np.argmax(y_train,axis=1)[...,np.newaxis])
                    w_c = w_c.astype('float32')
                    c_c = c_c.astype('float32')
                else:
                    w_c, c_c = 0, 0
                out.extend([w_c,c_c])

            tf.keras.backend.clear_session()
    return out

def test_models(x_test_cnn, y_test, x_lda, y_lda, cnn=None, lda=None, clda=None, test_mod=None, test_accuracy=None):
    acc = np.empty((2,))
    acc[:] = np.nan

    #test LDA
    if lda is not None:
        w = lda[0]
        c = lda[1]
        # acc[0],out = eval_lda(w, c, x_lda, y_lda) * 100
        acc[0] = eval_lda(w, c, x_lda, y_lda) *100
    
    # test CNN
    if cnn is not None:
        if clda is None:
            if test_mod is None:
                test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
                test_mod = get_test(cnn, test_accuracy)
            test_accuracy.reset_states()
            test_mod(tf.convert_to_tensor(x_test_cnn), tf.convert_to_tensor(y_test))
            acc[1] = test_accuracy.result()*100
        else:
            w = clda[0]
            c = clda[1]
            acc[1] = eval_lda(w, c, cnn.enc(x_test_cnn).numpy(), np.argmax(y_test,axis=1)[...,np.newaxis]) * 100

    return acc

def check_labels(test_data,test_params,train_dof,key,test_key=True):

    for dof in train_dof:
        test_params[test_params[:,-1]==dof,0] = key[train_dof==dof]

    if test_key:
        test_dof = np.unique(test_params[:,-1])
        print('init test dof: ' + str(test_dof))
        xtra_dof = ~np.isin(test_dof,train_dof)
        for dof in test_dof[xtra_dof]:
            print('removing extra test DOF ' + str(dof))
            ind = test_params[:,-1] == dof
            test_params = test_params[~ind,...]
            test_data = test_data[~ind,...]
        test_dof = np.delete(test_dof,xtra_dof)
        key2 = np.zeros(train_dof.shape)
        key2[:] = np.nan
        for dof in train_dof:
            ind = test_params[:,-1] == dof
            if np.sum(ind) > 0:
                key2[train_dof==dof] = np.nanmean(test_params[ind,0])
    
        print('test_dof: ' + str(test_dof) + ', key: ' + str(key2))

    return test_data, test_params



        