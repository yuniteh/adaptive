import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model
import numpy as np
import copy as cp

## Encoders
class MLPenc(Model):
    def __init__(self, latent_dim=4, name='enc'):
        super(MLPenc, self).__init__(name=name)
        self.dense1 = Dense(246, activation='relu')
        self.bn1 = BatchNormalization()
        self.dense2 = Dense(128, activation='relu')
        self.bn2 = BatchNormalization()
        self.dense3 = Dense(16, activation='relu')
        self.bn3 = BatchNormalization()
        self.latent = Dense(latent_dim, activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.bn4 = BatchNormalization()

    def call(self, x):
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dense3(x)
        x = self.bn3(x)
        x = self.latent(x)
        return self.bn4(x)

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
    def __init__(self, n_class=7, name='clf'):
        super(CLF, self).__init__(name=name)
        self.dense1 = Dense(n_class, activation='softmax')

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
    def __init__(self):
        super(EWC, self).__init__()
        self.enc = MLPenc()
        self.clf = CLF()
    
    def call(self, x):
        x = self.enc(x)
        return self.clf(x)

    def compute_diag_fim(self, x_data, fim_samples=1000):
        '''
            compute diagonal fisher information matrix
        '''
        FIM = []
        for ii in range(len(self.trainable_weights)):
            FIM.append(np.zeros(self.trainable_weights[ii].shape))
        # -- set-up graph nodes--
        # true fisher information: use predicted label
        # get gradients wrt log likelihood
        
        # -- compute FIM (expected squared score of ll) --
        for i in range(fim_samples):
            idx = np.random.randint(x_data.shape[0])
            with tf.GradientTape() as tape:
                self.y = self.call(x_data[idx:idx+1,:])
                c_index = tf.argmax(self.y,1)[0]
                loss = tf.math.log(self.y[0,c_index])
            grads = tape.gradient(loss,self.trainable_weights)
            # add squared gradients of score:
            for ii in range(len(self.trainable_weights)):
                FIM[ii] +=  grads[ii]**2 # np.square(grads[ii])
        # normalise:
        self.FIM  = [FIM[ii]/fim_samples for ii in range(len(self.trainable_weights))]

    def star(self,mod=None):
        # used for saving optimal weights after most recent task training
        self.set_weights(mod.get_weights())
        self.star_vars = []
        for i in range(len(mod.trainable_weights)):
            self.star_vars.append(np.zeros(self.trainable_weights[i].shape))
            self.star_vars[i] = mod.trainable_weights[i].numpy()
        # for i in range(len(self.star_vars)):
        #     self.star_vars[i] = mod.trainable_weights[i].numpy()

    def update_ewc_loss(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints

        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.cross_entropy

        for v in range(len(self.var_list)):
            self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))


def eval_nn(x, y, mod, clean):
    y_pred = np.argmax(mod(x).numpy(),axis=1)
    clean_acc = np.sum(y_pred[:clean] == np.argmax(y[:clean,...],axis=1))/y_pred[:clean].size
    noise_acc = np.sum(y_pred[clean:] == np.argmax(y[clean:,...],axis=1))/y_pred[clean:].size
    return clean_acc, noise_acc

## TRAIN TEST MLP
def get_train_ewc():
    @tf.function
    def train_step(x, y, mod, optimizer, train_loss, train_loss2, train_accuracy, lam = 0):
        with tf.GradientTape() as tape:
            y_out = mod(x,training=True)
            loss = tf.keras.losses.categorical_crossentropy(y,y_out)
            class_loss = tf.keras.losses.categorical_crossentropy(y,y_out)
            for v in range(len(mod.trainable_variables)):
                # print(v)
                # print(tf.reduce_sum(tf.multiply(mod.FIM[v],tf.square(mod.trainable_weights[v] - mod.star_vars[v]))))
                loss += (lam/2) * tf.reduce_sum(tf.multiply(mod.FIM[v],tf.square(mod.trainable_weights[v] - mod.star_vars[v])))
        # print(train_loss(loss).result())
        gradients = tape.gradient(loss,mod.trainable_weights)
        optimizer.apply_gradients(zip(gradients, mod.trainable_variables))

        train_loss(loss)
        train_loss2(class_loss)
        train_accuracy(y, y_out)
    
    return train_step

def get_train(prop=False):
    @tf.function
    def train_step(x, y, mod, optimizer, train_loss, train_accuracy, train_prop_accuracy=0, y_prop=0, align=None):
        with tf.GradientTape() as tape:
            if prop:
                y_out, prop_out = mod(x,training=True)
                class_loss = tf.keras.losses.categorical_crossentropy(y,y_out)
                prop_loss = tf.keras.losses.mean_squared_error(y_prop, prop_out)
                loss = class_loss + prop_loss/10
            else:
                if isinstance(align,ALI):
                    y_out = mod(align(x,training=True),training=False)
                else:
                    y_out = mod(x,training=True)
                loss = tf.keras.losses.categorical_crossentropy(y,y_out)
            
        gradients = tape.gradient(loss, mod.trainable_variables)
        optimizer.apply_gradients(zip(gradients, mod.trainable_variables))

        train_loss(loss)
        train_accuracy(y, y_out)
        if prop:
            train_prop_accuracy(y_prop, prop_out)
    
    return train_step

def get_test():
    @tf.function
    def test_step(x, y, mod, test_loss, test_accuracy,align=None):
        if isinstance(align,ALI):
            y_out = mod(align(x))
        else:
            y_out = mod(x)
        t_loss = tf.keras.losses.categorical_crossentropy(y,y_out)

        test_loss(t_loss)
        test_accuracy(y, y_out)
    
    return test_step
