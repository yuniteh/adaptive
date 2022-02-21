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

# variable initialization functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class EWC(Model):
    def __init__(self, x, y_, n_class=7):
        super(EWC, self).__init__()
        self.enc = MLPenc()
        self.clf = CLF(n_class)

        self.x = x # input placeholder
        self.y = self.call(x)

        self.var_list = self.get_weights()

        # vanilla single-task loss
        self.cross_entropy =  tf.keras.losses.categorical_crossentropy(self.y,y_)
        self.set_vanilla_loss()

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def call(self, x):
        x = self.enc(x)
        return self.clf(x)

    def compute_fisher(self, imgset, sess, num_samples=200):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        fish_gra = tf.gradients(tf.log(probs[0,class_ind]), self.var_list)
        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = sess.run(fish_gra, feed_dict={self.x: imgset[im_ind:im_ind+1]})
            # square the derivatives and add to total
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss(self):
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)

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
    def train_step(x, y, mod, optimizer, train_loss, train_accuracy, ewc=True, lams = 0):
        with tf.GradientTape() as tape:
            if(lams == 0):
                mod.set_vanilla_loss()
            else:
                mod.update_ewc_loss(lams)
            if ewc:
                y_out = mod(x,training=True)
                loss = mod.set_vanilla_loss
            else:
                y_out = mod(x,training=True)
                loss = tf.keras.losses.categorical_crossentropy(y,y_out)
            
        gradients = tape.gradient(loss, mod.trainable_variables)
        optimizer.apply_gradients(zip(gradients, mod.trainable_variables))

        train_loss(loss)
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
