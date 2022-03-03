import numpy as np
from numpy.linalg import eig, inv
import tensorflow as tf


# train and predict for data: (samples,feat), label: (samples, 1)
def eval_lda(w, c, x_test, y_test, clean_size=None):
    if clean_size:
        clean_out = predict(x_test[:clean_size,...], w, c)
        noise_out = predict(x_test[clean_size:,...], w, c)
        clean_acc = np.sum(clean_out.reshape(y_test[:clean_size,...].shape) == y_test[:clean_size,...])/y_test[:clean_size,...].shape[0]
        noise_acc = np.sum(noise_out.reshape(y_test[clean_size:,...].shape) == y_test[clean_size:,...])/y_test[clean_size:,...].shape[0]

        return clean_acc, noise_acc
    else:
        out = predict(x_test, w, c)
        acc = np.sum(out.reshape(y_test.shape) == y_test)/y_test.shape[0]
        
        return acc

# train LDA classifier for data: (samples,feat), label: (samples, 1)
def train_lda(data,label,mu_bool=False, mu_class = 0, C = 0):
    if not isinstance(data,np.ndarray):
        data = data.numpy()
    m = data.shape[1]
    u_class = np.unique(label)
    n_class = u_class.shape[0]

    if not mu_bool:
        mu = np.mean(data,axis=0,keepdims = True)
        C = np.zeros([m,m])
        mu_class = np.zeros([n_class,m])
        Sb = np.zeros([mu.shape[1],mu.shape[1]])
        Sw = np.zeros([mu.shape[1],mu.shape[1]])

        for i in range(0,n_class):
            ind = label == u_class[i]
            mu_class[i,:] = np.mean(data[ind[:,0],:],axis=0,keepdims=True)
            C += np.cov(data[ind[:,0],:].T)
            Sb += ind.shape[0] * np.dot((mu_class[np.newaxis,i,:] - mu).T,(mu_class[np.newaxis,i,:] - mu)) 

            Sw_temp = np.zeros([mu.shape[1],mu.shape[1]])
            for row in data[ind[:,0],:]:
                Sw_temp += np.dot((row[:,np.newaxis] - mu_class[i,:,np.newaxis]), (row[:,np.newaxis] - mu_class[i,:,np.newaxis]).T)
            Sw += Sw_temp
        C /= n_class
        u,v = eig(inv(Sw).dot(Sb))    
        v = v[:,np.flip(np.argsort(np.abs(u)))]
        v = v[:,:6].real

    prior = 1/n_class

    w = np.zeros([n_class, m])
    c = np.zeros([n_class, 1])

    for i in range(0, n_class):
        w[i,:] = np.dot(mu_class[np.newaxis,i,:],np.linalg.pinv(C))
        c[i,:] = np.dot(-.5 * np.dot(mu_class[np.newaxis,i,:], np.linalg.pinv(C)),mu_class[np.newaxis,i,:].T) + np.log(prior)    

    if not mu_bool:
        return w, c, mu_class, C, v
    else:
        return w, c

def predict(data,w,c):
    f = np.dot(w,data.T) + c
    out = np.argmax(f, axis=0)
    return out

def predict_tf(data,w,c):
    f = tf.matmul(w,tf.transpose(data)) + c
    out = tf.argmax(f, axis=0)
    return out
