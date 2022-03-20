import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Conv2DTranspose, Reshape, Concatenate
from tensorflow.keras import Model
import numpy as np
import copy as cp
import matplotlib.pyplot as plt

class VAR(Model):
    def __init__(self, latent_dim=16, c1=32, c2=32, name='var'):
        super(VAR,self).__init__(name=name)
        self.conv1 = Conv2D(c1,3, activation='relu', strides=1, padding="same", activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn1 = BatchNormalization()#renorm=True)
        self.conv2 = Conv2D(c2,3, activation='relu', strides=1, padding="same", activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn2 = BatchNormalization()#renorm=True)
        self.flatten = Flatten(dtype="float32")
    
    def call(self, x, bn_training=False, bn_trainable=False):
        self.bn1.trainable=bn_trainable
        self.bn2.trainable=bn_trainable
        x = self.conv1(x)
        x = self.bn1(x,training=bn_training)
        x = self.conv2(x)
        x = self.bn2(x,training=bn_training)
        x = self.flatten(x)
        return x
    
    def get_shapes(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        conv2_s = tf.shape(x)
        x = self.flatten(x)
        flat_s = tf.shape(x)
        return flat_s, conv2_s

class DEC(Model):
    def __init__(self, flat_s, conv2_s, latent_dim=16,name='dec'):
        super(DEC,self).__init__(name=name)
        self.den = Dense(32, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn = BatchNormalization()#renorm=True)
        self.mean = Dense(latent_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.logvar = Dense(latent_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5),kernel_initializer='zeros',bias_initializer='zeros')
        self.vbn1 = BatchNormalization(dtype="float32")
        self.vbn2 = BatchNormalization(dtype="float32")
        self.cat = Concatenate()
        self.den1 = Dense(32, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn1 = BatchNormalization()
        self.den2 = Dense(flat_s[1], activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn2 = BatchNormalization()
        self.conv2_s = conv2_s[1:].numpy().tolist()
        self.tconv = Conv2DTranspose(32, 3, activation='relu', padding='same')
        self.bn3 = BatchNormalization()
        self.tconv2 = Conv2DTranspose(1, 3, activation='sigmoid', padding='same',dtype="float32")

    def call(self, x, cls, samp=False, bn_training=False, bn_trainable=False):
        if samp:
            x = self.sampling(x)
            z_mean = 0
            z_logvar = 0
        else:
            x = self.den(x)
            x = self.bn(x)
            z_mean = self.mean(x)
            z_logvar = self.logvar(x)
            z_mean = self.vbn1(z_mean)
            z_logvar = self.vbn2(z_logvar)
            x = self.sampling([z_mean, z_logvar])

        x2 = tf.cast(tf.tile(cls[...,tf.newaxis],[1,x.shape[1]]),x.dtype)
        self.bn1.trainable = bn_trainable
        self.bn2.trainable = bn_trainable
        self.bn3.trainable = bn_trainable
        x = self.cat([x,x2])
        x = self.den1(x)
        x = self.bn1(x,training = bn_training)
        x = self.den2(x)
        x = self.bn2(x,training = bn_training)
        x = tf.reshape(x,[x.shape[0]]+self.conv2_s)
        x = self.tconv(x)
        x = self.bn3(x,training = bn_training)
        x = self.tconv2(x)
        return x, z_mean, z_logvar

    def sampling(self, x):
        #Reparameterization trick by sampling from an isotropic unit Gaussian.
        if isinstance(x,list):
            z_mean, z_log_var = x
        else:
            z_mean = tf.zeros(tf.shape(x))
            z_log_var = tf.zeros(tf.shape(x))

        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim), dtype=z_mean.dtype)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

class CNNenc(Model):
    def __init__(self, latent_dim=16, c1=32, c2=32,name='enc'):
        super(CNNenc, self).__init__(name=name)
        self.conv1 = Conv2D(c1,3, activation='relu', strides=1, padding="same", activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn1 = BatchNormalization()#renorm=True)
        self.conv2 = Conv2D(c2,3, activation='relu', strides=1, padding="same", activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn2 = BatchNormalization()#renorm=True)
        self.flatten = Flatten()
        self.dense1 = Dense(32, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn3 = BatchNormalization()#renorm=True)
        self.latent = Dense(latent_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
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
    def __init__(self, latent_dim=8, c2=32, name='enc'):
        super(CNNbase, self).__init__(name=name)
        self.conv2 = Conv2D(c2,3, activation='relu', strides=1, padding="same")
        self.bn2 = BatchNormalization()
        self.flatten = Flatten()
        self.dense1 = Dense(16, activation='relu')
        self.bn3 = BatchNormalization()
        self.latent = Dense(latent_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
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
class VCLF(Model):
    def __init__(self, n_class=7, latent_dim=8, act='softmax', name='clf'):
        super(VCLF, self).__init__(name=name)
        self.dense1 = Dense(32, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn1 = BatchNormalization()
        self.latent = Dense(latent_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn2 = BatchNormalization()
        self.dense2 = Dense(n_class, activation=act, activity_regularizer=tf.keras.regularizers.l1(10e-5),dtype="float32")

    def call(self, x, bn_training=False, bn_trainable=False):
        self.bn1.trainable = bn_trainable
        self.bn2.trainable = bn_trainable
        x = self.dense1(x)
        x = self.bn1(x,training = bn_training)
        x = self.latent(x)
        x = self.bn2(x,training = bn_training)
        return self.dense2(x)

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
        self.enc = VAR(c1=c1,c2=c2)
        self.clf = VCLF(n_class)
    
    def add_dec(self, x):
        if not hasattr(self,'dec'):
            flat_s, conv2_s = self.enc.get_shapes(x)
            self.dec = DEC(flat_s, conv2_s)
    
    def call(self, x, y=None, bn_training=False, bn_trainable=False, dec=False):
        x = self.enc(x,bn_training=bn_training,bn_trainable=bn_trainable)
        y_out = self.clf(x,bn_training=bn_training,bn_trainable=bn_trainable)
        if dec:
            x_out, z_mean, z_logvar = self.dec(x, y, bn_training=bn_training,bn_trainable=bn_trainable)
            return [y_out, x_out, z_mean, z_logvar]
        else:
            return [y_out]

class CNN(Model):
    def __init__(self, n_class=7, c1=32, c2=32, adapt=False):
        super(CNN, self).__init__()
        if adapt:
            self.top = CNNtop(c1=c1)
            self.base = CNNbase(c2=c2)
        else:
            self.enc = VAR(c1=c1,c2=c2)
            # self.enc = CNNenc(c1=c1,c2=c2)
        # self.clf = CLF(n_class)
        self.clf = VCLF(n_class)
    
    def call(self, x, bn_training=False, bn_trainable=False):
        if hasattr(self,'top'):
            x = self.top(x, bn_training=bn_training, bn_trainable=bn_trainable)
            x = self.base(x, bn_training=bn_training, bn_trainable=bn_trainable)
        else:
            x = self.enc(x, bn_training=bn_training, bn_trainable=bn_trainable)
        y = self.clf(x)
        return y

class EWC(Model):
    def __init__(self, n_class=7, adapt=False):
        super(EWC, self).__init__()
        if adapt:
            self.top = CNNtop()
            self.base = CNNbase()
        else:
            self.enc = VAR()
        self.clf = VCLF(n_class=n_class)
    
    def acc(self, x, y, val_acc=None):
        y_out = self.call(x)
        if val_acc is None:
            val_acc = tf.keras.metrics.CategoricalAccuracy(name='val_acc')
        val_acc(y, y_out)
        return val_acc.result()
    
    def call(self, x, bn_training=False, bn_trainable=False):
        if hasattr(self,'top'):
            x = self.top(x)
            x = self.base(x)
        else:
            x = self.enc(x, bn_training=bn_training, bn_trainable=bn_trainable)
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
            y_out = mod(x,training=False,bn_training=False,bn_trainable=False)
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
    def train_step(x, y, mod, optimizer, train_loss=None, sec_loss=None, third_loss=None, train_accuracy=None, adapt=False, lam=0, clda=None, trainable=True, dec=False):
        with tf.GradientTape() as tape:
            if isinstance(mod,VCNN):
                mod_out = mod(x, y=tf.argmax(y,axis=-1), training=True, bn_training=True, bn_trainable=trainable, dec=dec)
                y_out = mod_out[0]
                class_loss = tf.keras.losses.categorical_crossentropy(y,y_out)
                loss = class_loss 
                if hasattr(mod,'dec') and dec:
                    _, x_out, z_mean, z_log_var = mod_out
                    kl_loss = -.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),axis=-1)
                    rec_loss = K.sum(tf.keras.losses.mean_squared_error(x, x_out))
                    # rec_loss = K.mean(tf.keras.losses.binary_crossentropy(x, x_out))
                    loss = rec_loss*lam[0] + kl_loss*lam[1]
            else:
                if adapt:
                    y_out = mod(x,training=True, bn_training=True, bn_trainable=trainable)
                else:
                    if clda is not None:
                        mod.clf.trainable = False
                        y_out = tf.nn.softmax(tf.transpose(tf.matmul(tf.cast(clda[0],tf.float32),tf.transpose(mod.enc(x,training=True))) + tf.cast(clda[1],tf.float32)))
                    else:
                        y_out = mod(x,training=True,bn_training=True,bn_trainable=trainable)
                
                loss = tf.keras.losses.categorical_crossentropy(y,y_out)

                if isinstance(mod, EWC) and hasattr(mod, "F_accum"):
                    for v in range(len(mod.trainable_weights)):
                        f_loss_orig = tf.cast(tf.reduce_sum(tf.multiply(mod.F_accum[v].astype(np.float32),tf.square(mod.trainable_weights[v] - mod.star_vars[v]))), loss.dtype)
                        f_loss = tf.cast((lam/2), loss.dtype) * f_loss_orig
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
                if isinstance(mod,EWC) and lam > 0:
                    gradients,_ = tf.clip_by_global_norm(gradients,50000)
                optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, mod.trainable_variables) if grad is not None)

        if train_loss is not None:
            train_loss(loss)
        if sec_loss is not None:
            if hasattr(mod,"F_accum"):
                sec_loss(f_loss_orig)
            elif dec:
                sec_loss(rec_loss)
                third_loss(kl_loss)
        if train_accuracy is not None:
            train_accuracy(y, y_out)
    
    return train_step

def get_ewc():
    @tf.function
    def train_step(x, y, mod, optimizer, train_loss=None, sec_loss=None, train_accuracy=None, lam=0, clda=None, trainable=True):
        with tf.GradientTape() as tape:
            if clda is not None:
                mod.clf.trainable = False
                y_out = tf.nn.softmax(tf.transpose(tf.matmul(tf.cast(clda[0],tf.float32),tf.transpose(mod.enc(x,training=True))) + tf.cast(clda[1],tf.float32)))
            else:
                y_out = mod(x,training=True,train=trainable,bn_trainable=trainable)
            
            loss = tf.keras.losses.categorical_crossentropy(y,y_out)

            if hasattr(mod, "F_accum"):
                for v in range(len(mod.trainable_weights)):
                    f_loss_orig = tf.reduce_sum(tf.multiply(mod.F_accum[v].astype(np.float32),tf.square(mod.trainable_weights[v] - mod.star_vars[v])))
                    f_loss = tf.cast((lam/2) * tf.reduce_sum(tf.multiply(mod.F_accum[v].astype(np.float32),tf.square(mod.trainable_weights[v] - mod.star_vars[v]))),loss.dtype)
                    loss += f_loss             
        
        if clda is not None:
            gradients = tape.gradient(loss, mod.enc.trainable_variables)
            optimizer.apply_gradients(zip(gradients, mod.enc.trainable_variables))
        else:
            gradients = tape.gradient(loss, mod.trainable_variables)
            if isinstance(mod,EWC) and lam > 0:
                gradients,_ = tf.clip_by_global_norm(gradients,50000)
            optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, mod.trainable_variables) if grad is not None)

        if train_loss is not None:
            train_loss(loss)
        if sec_loss is not None and hasattr(mod,"F_accum"):
            sec_loss(f_loss_orig)
        if train_accuracy is not None:
            train_accuracy(y, y_out)
    
    return train_step

def get_cnn(mod, optimizer, train_loss=None, train_accuracy=None, adapt=False, clda=None, trainable=True):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            if adapt:
                y_out = mod(x,training=True, train=True, bn_trainable=trainable)
            else:
                if clda is not None:
                    mod.clf.trainable = False
                    y_out = tf.nn.softmax(tf.transpose(tf.matmul(tf.cast(clda[0],tf.float32),tf.transpose(mod.enc(x,training=True))) + tf.cast(clda[1],tf.float32)))
                else:
                    y_out = mod(x,training=True,train=trainable,bn_trainable=trainable)
            
            loss = tf.keras.losses.categorical_crossentropy(y,y_out)       
        
        if adapt:
            gradients = tape.gradient(loss, mod.trainable_variables)
            optimizer.apply_gradients(zip(gradients, mod.trainable_variables))
        else:
            if clda is not None:
                gradients = tape.gradient(loss, mod.enc.trainable_variables)
                optimizer.apply_gradients(zip(gradients, mod.enc.trainable_variables))
            else:
                gradients = tape.gradient(loss, mod.trainable_variables)
                optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, mod.trainable_variables) if grad is not None)

        if train_loss is not None:
            train_loss(loss)
        if train_accuracy is not None:
            train_accuracy(y, y_out)
    
    return train_step

def get_vcnn():
    @tf.function
    def train_step(x, y, mod, optimizer, train_loss=None, sec_loss=None, third_loss=None, train_accuracy=None, lam=[0,0], trainable=True, dec=False):
        with tf.GradientTape() as tape:
            mod_out = mod(x, y=tf.argmax(y,axis=-1), training=True, bn_training=True, bn_trainable=trainable, dec=dec)
            y_out = mod_out[0]
            loss = tf.keras.losses.categorical_crossentropy(y,y_out)
            if hasattr(mod,'dec') and dec:
                _, x_out, z_mean, z_log_var = mod_out
                kl_loss = -.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),axis=-1)
                rec_loss = K.mean(tf.keras.losses.mean_squared_error(x, x_out))
                loss = rec_loss*lam[0] + kl_loss*lam[1]    
        
        gradients = tape.gradient(loss, mod.trainable_variables)
        optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, mod.trainable_variables) if grad is not None)

        if train_loss is not None:
            train_loss(loss)
        if sec_loss is not None and dec:
            sec_loss(rec_loss)
            third_loss(kl_loss)
        if train_accuracy is not None:
            train_accuracy(y, y_out)
    
    return train_step

def get_test(mod, test_accuracy=None):
    @tf.function(experimental_relax_shapes=True)
    def test_step(x, y):
        if hasattr(mod, 'dec'):
            y_out = mod(x,training=False,bn_training=False,bn_trainable=False,dec=False)[0]
        else:
            y_out = mod(x,training=False,bn_training=False,bn_trainable=False)

        if test_accuracy is not None:
            test_accuracy(y, y_out)
    
    return test_step
