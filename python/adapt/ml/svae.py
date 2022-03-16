import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Conv2DTranspose, Reshape, Concatenate
from tensorflow.keras import Model
import numpy as np
import copy as cp
import matplotlib.pyplot as plt

class VAR(Model):
    def __init__(self, latent_dim=4, c1=32, c2=32, name='var'):
        super(VAR,self).__init__(name=name)
        self.conv1 = Conv2D(c1,3, activation='relu', strides=1, padding="same", activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn1 = BatchNormalization()#renorm=True)
        self.conv2 = Conv2D(c2,3, activation='relu', strides=1, padding="same", activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn2 = BatchNormalization()#renorm=True)
        self.flatten = Flatten(dtype="float32")
    
    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
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
    def __init__(self, flat_s, conv2_s, latent_dim=4,name='dec'):
        super(DEC,self).__init__(name=name)
        self.den = Dense(16, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn = BatchNormalization()#renorm=True)
        self.mean = Dense(latent_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.logvar = Dense(latent_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5),kernel_initializer='zeros',bias_initializer='zeros')
        self.vbn1 = BatchNormalization(dtype="float32")
        self.vbn2 = BatchNormalization(dtype="float32")
        self.cat = Concatenate()
        self.den1 = Dense(16, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn1 = BatchNormalization()
        self.den2 = Dense(flat_s[1], activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn2 = BatchNormalization()
        self.conv2_s = conv2_s[1:].numpy().tolist()
        self.tconv = Conv2DTranspose(32, 3, activation='relu', padding='same')
        self.bn3 = BatchNormalization()
        self.tconv2 = Conv2DTranspose(1, 3, activation='sigmoid', padding='same',dtype="float32")

    def call(self, x, cls, samp=False):
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
        x = self.cat([x,x2])
        x = self.den1(x)
        x = self.bn1(x)
        x = self.den2(x)
        x = self.bn2(x)
        x = tf.reshape(x,[x.shape[0]]+self.conv2_s)#self.rshape(x)
        x = self.tconv(x)
        x = self.bn3(x)
        x = self.tconv2(x)
        return x, z_mean, z_logvar
    
    def sample(self,x):
        batch = K.shape(x)[0]
        dim = K.int_shape(x)[1]
        epsilon = K.random_normal(shape=(batch, dim), dtype=x.dtype)
        return epsilon

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
    
## Classifier
class VCLF(Model):
    def __init__(self, n_class=7, act='softmax', name='clf'):
        super(VCLF, self).__init__(name=name)
        self.dense1 = Dense(16, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn1 = BatchNormalization()
        self.dense2 = Dense(n_class, activation=act, activity_regularizer=tf.keras.regularizers.l1(10e-5),dtype="float32")

    def call(self, x):
        x = self.dense1(x)
        x = self.bn1(x)
        return self.dense2(x)

class VCNN(Model):
    def __init__(self, n_class=7, c1=32, c2=32):
        super(VCNN, self).__init__()
        self.var = VAR(c1=c1,c2=c2)
        self.clf = VCLF(n_class)
    
    def add_dec(self, x):
        flat_s, conv2_s = self.var.get_shapes(x)
        self.dec = DEC(flat_s, conv2_s)
    
    def call(self, x, y=None, dec=False):
        x = self.var(x)
        y_out = self.clf(x)
        if dec:
            x_out, z_mean, z_logvar = self.dec(x, y)
            return [y_out, x_out, z_mean, z_logvar]
        else:
            return [y_out]

def train_vcnn(mod,optimizer, lam, dec=False):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            mod_out = mod(x,training=True, y=tf.argmax(y,axis=-1),dec=dec)
            y_out = mod_out[0]
            class_loss = tf.keras.losses.categorical_crossentropy(y,y_out)
            loss = class_loss 
            if hasattr(mod,'dec') and dec:
                _, x_out, z_mean, z_log_var = mod_out
                kl_loss = -.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),axis=-1)
                rec_loss = K.mean(tf.keras.losses.mean_squared_error(x, x_out))
                # rec_loss = K.mean(tf.keras.losses.binary_crossentropy(x, x_out))#*x.shape[1]*x.shape[2]
                loss = rec_loss*lam[0] + kl_loss*lam[1]

        gradients = tape.gradient(loss, mod.trainable_variables)
        optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, mod.trainable_variables) if grad is not None)
    return
    
