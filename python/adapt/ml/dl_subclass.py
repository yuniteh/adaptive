import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model
import numpy as np
import copy as cp
import matplotlib.pyplot as plt

class VAR(Model):
    def __init__(self, latent_dim=4, name='var'):
        super(VAR,self).__init__(name=name)
        self.mean = Dense(latent_dim, activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.logvar = Dense(latent_dim, activity_regularizer=tf.keras.regularizers.l1(10e-5),kernel_initializer='zeros',bias_initializer='zeros')
        self.vbn1 = BatchNormalization()
        self.vbn2 = BatchNormalization()
    
    def call(self, x):
        z_mean = self.mean(x)
        z_logvar = self.logvar(x)
        z_mean = self.vbn1(z_mean)
        z_log_var = self.vbn2(z_logvar)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_logvar, z

    def sampling(self, z_mean, z_log_var):
        #Reparameterization trick by sampling from an isotropic unit Gaussian.
        # z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim), dtype=z_mean.dtype)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

class CNNenc(Model):
    def __init__(self, latent_dim=4, c1=32, c2=32,name='enc'):
        super(CNNenc, self).__init__(name=name)
        self.conv1 = Conv2D(c1,3, activation='relu', strides=1, padding="same", activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn1 = BatchNormalization()#renorm=True)
        self.conv2 = Conv2D(c2,3, activation='relu', strides=1, padding="same", activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn2 = BatchNormalization()#renorm=True)
        self.flatten = Flatten()
        self.dense1 = Dense(16, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn3 = BatchNormalization()#renorm=True)
        self.latent = Dense(latent_dim, activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn4 = BatchNormalization()#renorm=True)

    def call(self, x, train=False, bn_trainable=False):
        self.bn1.trainable = bn_trainable
        self.bn2.trainable = bn_trainable
        self.bn3.trainable = bn_trainable
        self.bn4.trainable = bn_trainable
        x = self.conv1(x)
        x = self.bn1(x, training=train)
        x = self.conv2(x)
        x = self.bn2(x, training=train)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn3(x, training=train)
        x = self.latent(x)
        x = self.bn4(x, training=train)
        return x

class CNNbase(Model):
    def __init__(self, latent_dim=4, c2=32, name='enc'):
        super(CNNbase, self).__init__(name=name)
        self.conv2 = Conv2D(c2,3, activation='relu', strides=1, padding="same")
        self.bn2 = BatchNormalization()
        self.flatten = Flatten()
        self.dense1 = Dense(16, activation='relu')
        self.bn3 = BatchNormalization()
        self.latent = Dense(latent_dim, activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn4 = BatchNormalization()

    def call(self, x, train=False, bn_trainable=False):
        self.bn2.trainable = bn_trainable
        self.bn3.trainable = bn_trainable
        self.bn4.trainable = bn_trainable
        x = self.conv2(x)
        x = self.bn2(x, training=train)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn3(x, training=train)
        x = self.latent(x)
        x = self.bn4(x, training=train)
        return x

class CNNtop(Model):
    def __init__(self, c1=32, c2=32,name='enc'):
        super(CNNtop, self).__init__(name=name)
        self.conv1 = Conv2D(c1,3, activation='relu', strides=1, padding="same")
        self.bn1 = BatchNormalization()

    def call(self, x, train=False, bn_trainable=False):
        self.bn1.trainable = bn_trainable
        x = self.conv1(x)
        x = self.bn1(x, training=train)
        return x

## Classifier
class CLF(Model):
    def __init__(self, n_class=7, act='softmax', name='clf'):
        super(CLF, self).__init__(name=name)
        self.dense1 = Dense(n_class, activation=act, activity_regularizer=tf.keras.regularizers.l1(10e-5))

    def call(self, x):
        return self.dense1(x)

class VCNN(Model):
    def __init__(self, n_class=7, c1=32, c2=32):
        super(VCNN, self).__init__()
        self.enc = CNNenc(c1=c1,c2=c2)
        self.var = VAR()
        self.clf = CLF(n_class)
    
    def call(self, x, train=False, bn_trainable=False):
        x = self.enc(x)
        z_mean, z_logvar, z = self.var(x)
        y = self.clf(z)
        return y

class CNN(Model):
    def __init__(self, n_class=7, c1=32, c2=32, adapt=False):
        super(CNN, self).__init__()
        if adapt:
            self.top = CNNtop(c1=c1)
            self.base = CNNbase(c2=c2)
        else:
            self.enc = CNNenc(c1=c1,c2=c2)
        self.clf = CLF(n_class)
    
    def call(self, x, train=False, bn_trainable=False):
        if hasattr(self,'top'):
            x = self.top(x, train=train, bn_trainable=bn_trainable)
            x = self.base(x, train=train, bn_trainable=bn_trainable)
        else:
            x = self.enc(x, train=train, bn_trainable=bn_trainable)
        y = self.clf(x)
        return y

class EWC(Model):
    def __init__(self, n_class=7, adapt=False):
        super(EWC, self).__init__()
        if adapt:
            self.top = CNNtop()
            self.base = CNNbase()
        else:
            self.enc = CNNenc()
        self.clf = CLF(n_class=n_class)
    
    def acc(self, x, y, val_acc=None):
        y_out = self.call(x)
        if val_acc is None:
            val_acc = tf.keras.metrics.CategoricalAccuracy(name='val_acc')
        val_acc(y, y_out)
        return val_acc.result()
    
    def call(self, x, train=False, bn_trainable=False):
        if hasattr(self,'top'):
            x = self.top(x)
            x = self.base(x)
        else:
            x = self.enc(x, train=train, bn_trainable=bn_trainable)
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

        if plot_diffs:
            plt.plot(range(0, num_samples-disp_freq, disp_freq), mean_diffs)
            plt.xlabel("Number of samples")
            plt.ylabel("Mean absolute Fisher difference")

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples
        
        if not hasattr(self,"F_old"):
            self.F_old = cp.deepcopy(self.F_accum)
            self.int = 1
        else:
            for vi in range(len(self.F_accum)):
                self.F_accum[vi] = self.F_accum[vi] + self.F_old[vi]
            self.F_old = cp.deepcopy(self.F_accum)

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []
        self.all_vars = []

        for v in range(len(self.trainable_weights)):
            self.star_vars.append(cp.deepcopy(self.trainable_weights[v].numpy()))
        
        for v in range(len(self.non_trainable_weights)):
            self.all_vars.append(cp.deepcopy(self.non_trainable_weights[v].numpy()))

    def restore(self):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.trainable_weights)):
                self.trainable_weights[v].assign(self.star_vars[v])
            
            for v in range(len(self.non_trainable_weights)):
                self.non_trainable_weights[v].assign(self.all_vars[v])

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
            y_out = mod(x,training=False,train=False,bn_trainable=False)
            c_index = tf.argmax(y_out,1)[0]
            if y is None:
                loss = -tf.math.log(y_out[0,c_index])
            else:
                loss = tf.keras.losses.categorical_crossentropy(y,y_out)

        gradients = tape.gradient(loss,mod.trainable_weights)
        return gradients
    return train_fish

def get_train():
    @tf.function
    def train_step(x, y, mod, optimizer, train_loss=None, fish_loss=None, train_accuracy=None, train_prop_accuracy=None, y_prop=None, adapt=False, prop=False, lam=0, clda=None, trainable=True, vae=False):
        with tf.GradientTape() as tape:
            if prop:
                y_out, prop_out = mod(x,training=True)
                class_loss = tf.keras.losses.categorical_crossentropy(y,y_out)
                prop_loss = tf.keras.losses.mean_squared_error(y_prop,prop_out)
                loss = class_loss + prop_loss/10
            elif vae:
                y_out = mod(x,training=True)
                z_mean, z_log_var, z = mod.var(mod.enc(x,training=True), training=True)
                class_loss = tf.keras.losses.categorical_crossentropy(y,y_out)
                kl_loss = -.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),axis=-1)
                loss = class_loss + kl_loss
            else:
                if adapt:
                    # y_out = mod.clf(mod.base(mod.top(x,training=True, trainable=False),training=False, trainable=False),training=False)
                    y_out = mod(x,training=True,train=True,bn_trainable=trainable)
                else:
                    if clda is not None:
                        mod.clf.trainable = False
                        y_out = tf.nn.softmax(tf.transpose(tf.matmul(tf.cast(clda[0],tf.float32),tf.transpose(mod.enc(x,training=True))) + tf.cast(clda[1],tf.float32)))
                    else:
                        y_out = mod(x,training=True,train=trainable,bn_trainable=trainable)
                
                loss = tf.keras.losses.categorical_crossentropy(y,y_out)

                if isinstance(mod, EWC) and hasattr(mod, "F_accum"):
                    for v in range(len(mod.trainable_weights)):
                        f_loss_orig = tf.reduce_sum(tf.multiply(mod.F_accum[v].astype(np.float32),tf.square(mod.trainable_weights[v] - mod.star_vars[v])))
                        f_loss = tf.cast((lam/2) * tf.reduce_sum(tf.multiply(mod.F_accum[v].astype(np.float32),tf.square(mod.trainable_weights[v] - mod.star_vars[v]))),loss.dtype)
                        loss += f_loss             
        
        if adapt:
            gradients = tape.gradient(loss, mod.trainable_variables)
            optimizer.apply_gradients(zip(gradients, mod.trainable_variables))
        else:
            if clda is not None:
                gradients = tape.gradient(loss, mod.enc.trainable_variables)
                optimizer.apply_gradients(zip(gradients, mod.enc.trainable_variables))
            else:
                gradients = tape.gradient(loss, mod.trainable_variables)
                if lam > 0:
                    gradients,_ = tf.clip_by_global_norm(gradients,50000)
                optimizer.apply_gradients(zip(gradients, mod.trainable_variables))

        if train_loss is not None:
            train_loss(loss)
        if fish_loss is not None and hasattr(mod,"F_accum"):
            fish_loss(f_loss_orig)
        if train_accuracy is not None:
            train_accuracy(y, y_out)
        if train_prop_accuracy is not None:
            train_prop_accuracy(y_prop, prop_out)
    
    return train_step

def get_test():
    @tf.function
    def test_step(x, y, mod, test_loss=None, test_accuracy=None):
        y_out = mod(x,training=False,train=False,bn_trainable=False)
        loss = tf.keras.losses.categorical_crossentropy(y,y_out)

        if test_loss is not None:
            test_loss(loss)
        if test_accuracy is not None:
            test_accuracy(y, y_out)
    
    return test_step
