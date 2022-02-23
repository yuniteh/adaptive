import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from IPython import display

## Encoders
class MLPenc(Model):
    def __init__(self, latent_dim=4, name='enc'):
        super(MLPenc, self).__init__(name=name)
        self.dense1 = Dense(246, activation='relu')
        # self.bn1 = BatchNormalization()
        self.dense2 = Dense(128, activation='relu')
        # self.bn2 = BatchNormalization()
        self.dense3 = Dense(16, activation='relu')
        # self.bn3 = BatchNormalization()
        self.latent = Dense(latent_dim, activity_regularizer=tf.keras.regularizers.l1(10e-5))
        # self.bn4 = BatchNormalization()

    def call(self, x):
        x = self.dense1(x)
        # x = self.bn1(x)
        x = self.dense2(x)
        # x = self.bn2(x)
        x = self.dense3(x)
        # x = self.bn3(x)
        x = self.latent(x)
        # x = self.bn4(x)
        return x

class CNNenc(Model):
    def __init__(self, latent_dim=4, c1=32, c2=32,name='enc'):
        super(CNNenc, self).__init__(name=name)
        self.conv1 = Conv2D(c1,3, activation='relu', strides=1, padding="same")
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(c2,3, activation='relu', strides=1, padding="same")
        self.bn2 = BatchNormalization()
        self.flatten = Flatten()
        self.dense1 = Dense(16, activation='relu')
        self.bn3 = BatchNormalization()
        self.latent = Dense(latent_dim, activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn4 = BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn3(x)
        x = self.latent(x)
        return self.bn4(x)

## Classifier
class CLF(Model):
    def __init__(self, n_class=7, act='softmax', name='clf'):
        super(CLF, self).__init__(name=name)
        self.dense1 = Dense(n_class, activation=act)

    def call(self, x):
        return self.dense1(x)

class PROP(Model):
    def __init__(self, n_class=1, name='prop'):
        super(PROP, self).__init__(name=name)
        self.dense1 = Dense(n_class, activation='relu')

    def call(self, x):
        return self.dense1(x)

class MLP(Model):
    def __init__(self, n_class=7):
        super(MLP, self).__init__()
        self.enc = MLPenc()
        self.clf = CLF(n_class)
    
    def call(self, x):
        x = self.enc(x)
        return self.clf(x)

## Aligner
class ALI(Model):
    def __init__(self, n=32, name='aligner'):
        super(ALI, self).__init__(name=name)
        self.dense1 = Dense(64, activation='relu')
        self.bn1 = BatchNormalization()
        self.dense2 = Dense(n, activation='relu')

    def call(self,x):
        in_shape = x.shape
        x = tf.reshape(x,(x.shape[0],-1))
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dense2(x)
        if len(in_shape) > 2:
            x = tf.reshape(x,in_shape)
        return x

## Full models
class MLPprop(Model):
    def __init__(self, n_class=7, n_prop=1):
        super(MLPprop, self).__init__()
        self.enc = MLPenc()
        self.clf = CLF(n_class)
        self.prop = PROP(n_prop)
    
    def call(self, x):
        x = self.enc(x)
        y = self.clf(x)
        prop = self.prop(x)
        return y, prop

class CNN(Model):
    def __init__(self, n_class=7, c1=32, c2=32):
        super(CNN, self).__init__()
        self.enc = CNNenc(c1=c1,c2=c2)
        self.clf = CLF(n_class)
    
    def call(self, x):
        x = self.enc(x)
        y = self.clf(x)
        return y
  
class CNNprop(Model):
    def __init__(self, n_class=7, c1=32, c2=32):
        super(CNNprop, self).__init__() 
        self.enc = CNNenc(c1=c1,c2=c2)
        self.clf = CLF(n_class)
        self.prop = PROP(n_class)
    
    def call(self, x):
        x = self.enc(x)
        y = self.clf(x)
        prop = self.prop(x)
        return y, prop

class EWC(Model):
    def __init__(self, n_class=7):
        super(EWC, self).__init__()
        self.enc = MLPenc()
        self.clf = CLF(n_class=n_class)
    
    def acc(self, x, y, val_acc=None):
        y_out = self.call(x)
        if val_acc is None:
            val_acc = tf.keras.metrics.CategoricalAccuracy(name='val_acc')
        val_acc(y, y_out)
        return val_acc.result()
    
    def call(self, x):
        x = self.enc(x)
        return self.clf(x)

    def compute_fisher(self, imgset, y, num_samples=200, plot_diffs=False, disp_freq=10):
        # computer Fisher information for each parameter
        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.trainable_weights)):
            self.F_accum.append(np.zeros(self.trainable_weights[v].get_shape().as_list()))


        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = cp.deepcopy(self.F_accum)
            mean_diffs = np.zeros(0)

        fish_gra = get_fish()
        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = fish_gra(imgset[im_ind:im_ind+1],y[im_ind:im_ind+1],self)
            # square the derivatives and add to total
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(self.F_accum)):
                        F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for v in range(len(self.F_accum)):
                        F_prev[v] = self.F_accum[v]/(i+1)

        plt.plot(range(0, num_samples-disp_freq, disp_freq), mean_diffs)
        plt.xlabel("Number of samples")
        plt.ylabel("Mean absolute Fisher difference")
        display.display(plt.gcf())
        display.clear_output(wait=True)

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples
        
        if not hasattr(self,"F_old"):
            self.F_old = cp.deepcopy(self.F_accum)
            self.int = 1
        else:  
            self.int += 1
            for v in range(len(self.F_accum)):
                self.F_accum[v] = (self.F_accum[v] + self.F_old[v])/self.int
            self.F_old = cp.deepcopy(self.F_accum)


    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.trainable_weights)):
            self.star_vars.append(cp.deepcopy(self.trainable_weights[v].numpy()))

    def restore(self):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.trainable_weights)):
                self.trainable_weights[v].assign(self.star_vars[v])

def eval_nn(x, y, mod, clean):
    y_pred = np.argmax(mod(x).numpy(),axis=1)
    clean_acc = np.sum(y_pred[:clean] == np.argmax(y[:clean,...],axis=1))/y_pred[:clean].size
    noise_acc = np.sum(y_pred[clean:] == np.argmax(y[clean:,...],axis=1))/y_pred[clean:].size
    return clean_acc, noise_acc

## TRAIN TEST MLP
def get_fish():
    @tf.function
    def train_fish(x, y, mod):
        with tf.GradientTape() as tape:
            y_out = mod(x,training=True)
            c_index = tf.argmax(y_out,1)[0]
            loss = tf.math.log(y_out[0,c_index])

        gradients = tape.gradient(loss,mod.trainable_weights)
        return gradients
    return train_fish

def get_train():
    @tf.function
    def train_step(x, y, mod, optimizer, train_loss=None, fish_loss=None, train_accuracy=None, train_prop_accuracy=None, y_prop=None, align=None, prop=False, lam=0):
        with tf.GradientTape() as tape:
            if prop:
                y_out, prop_out = mod(x,training=True)
                class_loss = tf.keras.losses.categorical_crossentropy(y,y_out)
                prop_loss = tf.keras.losses.mean_squared_error(y_prop,prop_out)
                loss = class_loss + prop_loss/10
            else:
                if align is not None:
                    y_out = mod(align(x,training=True),training=False)
                else:
                    y_out = mod(x,training=True)
                
                loss = tf.keras.losses.categorical_crossentropy(y,y_out)

                if isinstance(mod, EWC) and hasattr(mod, "F_accum"):
                    for v in range(len(mod.trainable_weights)):
                        f_loss = (lam/2) * tf.reduce_sum(tf.multiply(mod.F_accum[v].astype(np.float32),tf.square(mod.trainable_weights[v] - mod.star_vars[v])))   
                        loss += (lam/2) * tf.reduce_sum(tf.multiply(mod.F_accum[v].astype(np.float32),tf.square(mod.trainable_weights[v] - mod.star_vars[v])))                    
            
        gradients = tape.gradient(loss, mod.trainable_variables)
        optimizer.apply_gradients(zip(gradients, mod.trainable_variables))

        if train_loss is not None:
            train_loss(loss)
        if fish_loss is not None and hasattr(mod,"F_accum"):
            fish_loss(f_loss)
        if train_accuracy is not None:
            train_accuracy(y, y_out)
        if train_prop_accuracy is not None:
            train_prop_accuracy(y_prop, prop_out)
    
    return train_step

def get_test():
    @tf.function
    def test_step(x, y, mod, test_loss=None, test_accuracy=None,align=None):
        if align is not None:
            y_out = mod(align(x))
        else:
            y_out = mod(x)
        loss = tf.keras.losses.categorical_crossentropy(y,y_out)

        if test_loss is not None:
            test_loss(loss)
        if test_accuracy is not None:
            test_accuracy(y, y_out)
    
    return test_step
