from re import A
import numpy as np
import scipy.io 
import pandas as pd
import copy as cp
import pickle
import os
from datetime import date
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations, product
import pickle
import tensorflow as tf
from adapt.ml.dl_subclass import get_test
from adapt.ml.lda import eval_lda

def load_raw(filename):
    struct = scipy.io.loadmat(filename)
    daq = struct['data'][0,0]['daq']['DAQ_DATA'][0,0]
    pvd = struct['data'][0,0]['pvd']

    return daq,pvd

# params structure: [subject ID, Iter, Index, Training group, DOF, Pos]
def load_train_data(filename):
    struct = scipy.io.loadmat(filename)

    feat = struct['feat']
    params = struct['params']
    daq = struct['daq']

    return feat,params,daq

def load_caps_train(filename):
    struct = scipy.io.loadmat(filename)
    data = struct['data_out']
    params = struct['params_out']
    
    return data, params

def process_daq(daq,params,win=200,ch=6):
    trial_data = np.zeros((win,6,params.shape[0]))
    for trial in range(0,params.shape[0]-1):
        sub = params[trial,0]
        grp = params[trial,1]
        ind = params[trial,2]
        trial_data[:,:,trial] = daq[sub-1,0][0,grp-1][ind-1:ind+win-1,:]
    
    return trial_data

def process_df(params):
    df = pd.DataFrame(data=params,columns=['sub','trial','ind','group','class','pos'])
    df = df.set_index('sub')
    
    return df

def load_noise_data(filename):
    struct = scipy.io.loadmat(filename)
    raw_noise = struct['raw_win']

    return raw_noise
    
def truncate(data):
    data[data > 5] = 5
    data[data < -5] = -5
    return data

def prep_train_caps(x_train, params, prop_b=True, num_classes=None, batch_size=128,noise=True, scaler=None, emg_scale=None, ft='feat', split=False):
    # x_train, params = threshold(x_train,params)
    # print(np.unique(params[:,-1]))
    if split:
        # for cl in np.unique(params[:,-1]):
        #     num = np.max(params[params[:,-1] == cl,1])
        #     if num 
        x_rest = x_train[params[:,-1] == 1,...]
        x_train_half = x_train[(params[:,1]%2==0) & (params[:,-1] != 1),...]
        x_train = np.vstack((x_rest[:x_rest.shape[0]//2,...],x_train_half))

        p_rest = params[params[:,-1] == 1,...]
        p_train_half = params[(params[:,1]%2==0) & (params[:,-1] != 1),...]
        params = np.vstack((p_rest[:p_rest.shape[0]//2,...],p_train_half))

    # print(np.unique(params[:,-1]))
    x_train, params = shuffle(x_train, params, random_state = 0)
    x_orig = cp.deepcopy(x_train)

    if not isinstance(emg_scale,np.ndarray):
        emg_scale = np.ones((np.size(x_train,1),1))
        for i in range(np.size(x_train,1)):
            emg_scale[i] = 5/np.max(np.abs(x_train[:,i,:]))
    x_train *= emg_scale

    if noise:
        x_train_noise, x_train_clean, y_train_clean = add_noise_caps(x_train, params, num_classes=num_classes)
            
        # shuffle data to make even batches
        x_train_noise, x_train_clean, y_train_clean = shuffle(x_train_noise, x_train_clean, y_train_clean, random_state = 0)
    else:
        y = to_categorical(params[:,0]-1,num_classes=num_classes)
        x_train_noise, y_train_clean = shuffle(x_train,y,random_state=0)
        x_train_clean = cp.deepcopy(x_train_noise)

    # calculate class MAV
    if prop_b:
        mav_all, _, _, _, y_train_clean, ind = extract_scale(x_train_clean,load=False,ft='mav',caps=True)
        mav_class = np.empty((y_train_clean.shape[1],x_train_clean.shape[1]))
        for i in range(mav_class.shape[0]):
            mav_class[i,:] = np.squeeze(np.mean(mav_all[y_train_clean[:,i].astype(bool),...],axis=0))
        mav_tot = np.sum(np.square(mav_class), axis=1)[...,np.newaxis]
        prop_temp = np.square((1 / mav_tot) * (mav_class @ np.squeeze(mav_all).T)).T
        prop = np.zeros(prop_temp.shape)
        prop[y_train_clean.astype(bool)] = prop_temp[y_train_clean.astype(bool)]        
    else:
        prop = np.empty((y_train_clean.shape))

    # Extract features
    if scaler is not None:
        load = True
    else:
        scaler = MinMaxScaler(feature_range=(0,1))
        load = False
        
    x_train_noise_cnn, scaler, x_min, x_max= extract_scale(x_train_noise,scaler=scaler,load=load,ft=ft,caps=True) 
    x_train_noise_cnn = x_train_noise_cnn.astype('float32')        
    print(scaler.get_params())

    # reshape data for nonconvolutional network
    x_train_noise_mlp = x_train_noise_cnn.reshape(x_train_noise_cnn.shape[0],-1)

    # create batches
    trainmlp = tf.data.Dataset.from_tensor_slices((x_train_noise_mlp, y_train_clean, prop)).shuffle(x_train_noise_mlp.shape[0],reshuffle_each_iteration=True).batch(batch_size)
    traincnn = tf.data.Dataset.from_tensor_slices((x_train_noise_cnn, y_train_clean, prop)).shuffle(x_train_noise_cnn.shape[0],reshuffle_each_iteration=True).batch(batch_size)

    # LDA data
    y_train_lda = params[:,[0]] - 1
    x_train_lda = extract_feats_caps(x_orig,ft=ft)

    return trainmlp, traincnn, y_train_clean, x_train_noise_mlp, x_train_noise_cnn, x_train_lda, y_train_lda, emg_scale, scaler, x_min, x_max, prop

def prep_test_caps(x, params, scaler=None, emg_scale=None, num_classes=None,ft='feat', split=False):
    # x = x[params[:,-1]!=1,...]
    # params = params[params[:,-1]!=1,...]
    # x, params = threshold(x,params)
    if split:
        x_rest = x[params[:,2] == 1,...]
        x_test_half = x[(params[:,1]%2!=0) & (params[:,2] !=1),...]
        x = np.vstack((x_rest[:x_rest.shape[0]//2,...],x_test_half))

        p_rest = params[params[:,2] == 1,...]
        p_test_half = params[(params[:,1]%2!=0) & (params[:,2] !=1),...]
        params = np.vstack((p_rest[:p_rest.shape[0]//2,...],p_test_half))

    x, params = shuffle(x, params, random_state = 0)
    y = to_categorical(params[:,0]-1,num_classes=num_classes)

    x_orig = cp.deepcopy(x)
    if emg_scale is not None:
        x *= emg_scale

    # shuffle data to make even batches
    x_test, y_test = shuffle(x, y, random_state = 0)
    x_test, y_test = x,y

    # Extract features
    if scaler is not None:
        x_test_cnn, _, _, _= extract_scale(x_test,scaler,load=True,ft=ft,caps=True) 
        x_test_cnn = x_test_cnn.astype('float32')

        # reshape data for nonconvolutional network
        x_test_mlp = x_test_cnn.reshape(x_test_cnn.shape[0],-1)
    else:
        x_test_cnn = 0
        x_test_mlp = 0

    # LDA data
    y_lda = params[:,[0]] - 1
    x_lda = extract_feats_caps(x_orig,ft=ft)

    return y_test, x_test_mlp, x_test_cnn, x_lda, y_lda

def add_noise_caps(raw, params,num_classes=None):
    all_ch = raw.shape[1]
    num_ch = 4
    split = 6
    rep = 2
    start_ch = 1
    sub_params = np.tile(params,(rep*(num_ch-1)+1,1))
    orig = np.tile(raw,(rep*(num_ch-1)+1,1,1))

    out = np.array([]).reshape(0,all_ch,200)
    x = np.linspace(0,0.2,200)

    # repeat twice if adding gauss and flat
    for rep_i in range(rep):   
        # loop through channel noise
        for num_noise in range(start_ch,num_ch):
            # find all combinations of noisy channels
            ch_all = list(combinations(range(0,all_ch),num_noise))
            temp = cp.deepcopy(raw)
            ch_split = temp.shape[0]//(split*len(ch_all))
            
            # loop through all channel combinations
            for ch in range(0,len(ch_all)):
                ch_noise = np.random.randint(3,size=(ch_split,num_noise))
                ch_level = np.random.randint(5,size=(ch_split,num_noise))

                if num_noise > 1:
                    for i in range(ch_split):
                        while np.array([x == ch_noise[i,0] for x in ch_noise[i,:]]).all() and np.array([x == ch_level[i,0] for x in ch_level[i,:]]).all():
                            ch_noise[i,:] = np.random.randint(3,size = num_noise)
                            ch_level[i,:] = np.random.randint(5,size = num_noise)

                ch_ind = 0
                for i in ch_all[ch]:       
                    if rep_i == 0:
                        temp[6*ch*ch_split:(6*ch+1)*ch_split,i,:] = 0
                        temp[(6*ch+2)*ch_split:(6*ch+3)*ch_split,i,:] += np.sin(2*np.pi*60*x)
                        temp[(6*ch+3)*ch_split:(6*ch+4)*ch_split,i,:] += 2*np.sin(2*np.pi*60*x)
                        temp[(6*ch+4)*ch_split:(6*ch+5)*ch_split,i,:] += 3*np.sin(2*np.pi*60*x) 
                        temp[(6*ch+5)*ch_split:(6*ch+6)*ch_split,i,:] += 4*np.sin(2*np.pi*60*x)
                        temp_split = temp[(6*ch+1)*ch_split:(6*ch+2)*ch_split,i,:]
                        for temp_iter in range(ch_split):
                            if ch_noise[temp_iter,ch_ind] == 0:
                                temp_split[temp_iter,...] = 0
                            elif ch_noise[temp_iter,ch_ind] == 1:
                                temp_split[temp_iter,...] += np.random.normal(0,ch_level[temp_iter,ch_ind]+1,temp.shape[2])
                            else:
                                temp_split[temp_iter,...] += (ch_level[temp_iter,ch_ind]+1)*np.sin(2*np.pi*60*x)
                        temp[(6*ch+1)*ch_split:(6*ch+2)*ch_split,i,:] = cp.deepcopy(temp_split)
                    else:        
                        temp[(6*ch)*ch_split:(6*ch+1)*ch_split,i,:] += 5*np.sin(2*np.pi*60*x)
                        temp[(6*ch+1)*ch_split:(6*ch+2)*ch_split,i,:] += np.random.normal(0,1,temp.shape[2])
                        temp[(6*ch+2)*ch_split:(6*ch+3)*ch_split,i,:] += np.random.normal(0,2,temp.shape[2])
                        temp[(6*ch+3)*ch_split:(6*ch+4)*ch_split,i,:] += np.random.normal(0,3,temp.shape[2])
                        temp[(6*ch+4)*ch_split:(6*ch+5)*ch_split,i,:] += np.random.normal(0,4,temp.shape[2])
                        temp[(6*ch+5)*ch_split:(6*ch+6)*ch_split,i,:] += np.random.normal(0,5,temp.shape[2])

                    ch_ind += 1 

            out = np.concatenate((out,temp))
    
    out = np.concatenate((raw, out))

    noisy, clean, y = out, orig, to_categorical(sub_params[:,0]-1,num_classes=num_classes)

    clean = clean[...,np.newaxis]
    noisy = noisy[...,np.newaxis]

    noisy = truncate(noisy)
    clean = truncate(clean)
    return noisy,clean,y

def add_noise(raw, params, n_type='flat', scale=5, real_noise=0,emg_scale=[1,1,1,1,1,1]):
    # Index subject and training group
    max_ch = raw.shape[1] + 1
    num_ch = int(n_type[-1]) + 1
    full_type = n_type[0:4]
    noise_type = n_type[4:-1]

    if noise_type[:3] == 'pos':
        num_ch = int(noise_type[-1]) + 1
        noise_type = noise_type[3:-1]
    
    if full_type == 'full':
        if noise_type == 'gaussflat60hz' or noise_type == 'allmix':
            split = 6
        elif noise_type == 'gauss60hz' or noise_type == '60hzall' or noise_type == 'gaussall' or noise_type == 'testall':
            split = 5
        else:
            split = 3
    else:
        split = 1

    rep = 1

    # tile data once for each channel
    if full_type == 'full':
        if noise_type != '60hzall' or noise_type != 'gaussall':
            rep = 2
        elif noise_type == 'testall':
            rep = 3
        start_ch = 1
        sub_params = np.tile(params,(rep*(num_ch-1)+1,1))
        orig = np.tile(raw,(rep*(num_ch-1)+1,1,1))
    # tile data twice, once for clean and once for noise
    elif full_type == 'part':
        start_ch = num_ch - 1
        sub_params = np.tile(params,(2,1))
        orig = np.tile(raw,(2,1,1))
        
    out = np.array([]).reshape(0,6,200)
    x = np.linspace(0,0.2,200)
    if noise_type == 'realmix':
        real_noise = np.delete(real_noise,(2),axis=0)
        # real_noise = np.delete(real_noise,(1),axis=0)
    elif noise_type == 'realmixnew' or noise_type == 'realmixeven':
        real_noise = np.delete(real_noise,(3),axis=0)
        # real_noise = np.delete(real_noise,(1),axis=0)
        real_type = real_noise.shape[0]

    # repeat twice if adding gauss and flat
    for rep_i in range(rep):   
        # loop through channel noise
        for num_noise in range(start_ch,num_ch):
            # find all combinations of noisy channels
            ch_all = list(combinations(range(0,6),num_noise))
            temp = cp.deepcopy(raw)
            ch_split = temp.shape[0]//(split*len(ch_all))
            
            # loop through all channel combinations
            for ch in range(0,len(ch_all)):
                if noise_type == 'mix' or noise_type == 'allmix':
                    ch_noise = np.random.randint(3,size=(ch_split,num_noise))
                    ch_level = np.random.randint(5,size=(ch_split,num_noise))

                    if num_noise > 1:
                        for i in range(ch_split):
                            while np.array([x == ch_noise[i,0] for x in ch_noise[i,:]]).all() and np.array([x == ch_level[i,0] for x in ch_level[i,:]]).all():
                                ch_noise[i,:] = np.random.randint(3,size = num_noise)
                                ch_level[i,:] = np.random.randint(5,size = num_noise)
                elif noise_type[:4] == 'real':
                    ch_noise = np.random.randint(1000,size=(ch_split,num_noise))
                    ch_level = np.random.randint(real_type,size=(ch_split,num_noise))
                    if noise_type == 'realmix':
                        if num_noise > 1:
                            for i in range(ch_split):
                                while np.array([x == ch_level[i,0] for x in ch_level[i,:]]).all():
                                    ch_level[i,:] = np.random.randint(real_type,size = num_noise)
                    elif noise_type == 'realmixeven':
                        noise_combo = np.array([x for x in product(np.arange(real_type),repeat=num_noise)])
                        rep_noise = ch_split//noise_combo.shape[0]
                        noise_all = np.tile(noise_combo,(rep_noise,1))
                        noise_extra = np.random.randint(real_type,size=(ch_split%noise_combo.shape[0],num_noise))
                        noise_all = np.concatenate((noise_all,noise_extra))
                    else:
                        ch_level = np.random.randint(real_type,size=(ch_split,num_noise))

                ch_ind = 0
                for i in ch_all[ch]:
                    if noise_type == '60hzall':
                        for scale_i in range(5):
                            temp[(5*ch+scale_i)*ch_split:(5*ch+scale_i+1)*ch_split,i,:] += (scale_i+1)*np.sin(2*np.pi*60*x)
                    elif noise_type == 'gaussall':
                        for scale_i in range(5):
                            temp[(5*ch+scale_i)*ch_split:(5*ch+scale_i+1)*ch_split,i,:] += np.random.normal(0,scale_i+1,temp.shape[2])
                    elif noise_type == 'gaussflat':
                        if rep_i == 0:
                            temp[3*ch*ch_split:(3*ch+1)*ch_split,i,:] = 0
                            temp[(3*ch+1)*ch_split:(3*ch+2)*ch_split,i,:] += np.random.normal(0,1,temp.shape[2])
                            temp[(3*ch+2)*ch_split:(3*ch+3)*ch_split,i,:] += np.random.normal(0,2,temp.shape[2])
                        else:
                            temp[3*ch*ch_split:(3*ch+1)*ch_split,i,:] += np.random.normal(0,3,temp.shape[2])
                            temp[(3*ch+1)*ch_split:(3*ch+2)*ch_split,i,:] += np.random.normal(0,4,temp.shape[2])
                            temp[(3*ch+2)*ch_split:(3*ch+3)*ch_split,i,:] += np.random.normal(0,5,temp.shape[2])
                    elif noise_type == 'flat60hz':
                        if rep_i == 0:
                            temp[3*ch*ch_split:(3*ch+1)*ch_split,i,:] = 0
                            temp[(3*ch+1)*ch_split:(3*ch+2)*ch_split,i,:] += np.sin(2*np.pi*60*x)
                            temp[(3*ch+2)*ch_split:(3*ch+3)*ch_split,i,:] += 2*np.sin(2*np.pi*60*x)
                        else:
                            temp[3*ch*ch_split:(3*ch+1)*ch_split,i,:] += 3*np.sin(2*np.pi*60*x)
                            temp[(3*ch+1)*ch_split:(3*ch+2)*ch_split,i,:] += 4*np.sin(2*np.pi*60*x)
                            temp[(3*ch+2)*ch_split:(3*ch+3)*ch_split,i,:] += 5*np.sin(2*np.pi*60*x)
                    elif noise_type == 'gauss60hz':
                        if rep_i == 0:
                            for scale_i in range(5):
                                temp[(5*ch+scale_i)*ch_split:(5*ch+scale_i+1)*ch_split,i,:] += (scale_i+1)*np.sin(2*np.pi*60*x)
                        else:        
                            for scale_i in range(5):
                                temp[(5*ch+scale_i)*ch_split:(5*ch+scale_i+1)*ch_split,i,:] += np.random.normal(0,scale_i+1,temp.shape[2])
                    elif noise_type == 'gaussflat60hz':
                        if rep_i == 0:
                            temp[6*ch*ch_split:(6*ch+2)*ch_split,i,:] = 0
                            temp[(6*ch+2)*ch_split:(6*ch+3)*ch_split,i,:] += np.sin(2*np.pi*60*x)
                            temp[(6*ch+3)*ch_split:(6*ch+4)*ch_split,i,:] += 2*np.sin(2*np.pi*60*x)
                            temp[(6*ch+4)*ch_split:(6*ch+5)*ch_split,i,:] += 3*np.sin(2*np.pi*60*x) 
                            temp[(6*ch+5)*ch_split:(6*ch+6)*ch_split,i,:] += 4*np.sin(2*np.pi*60*x)
                        else:        
                            temp[(6*ch)*ch_split:(6*ch+1)*ch_split,i,:] += 5*np.sin(2*np.pi*60*x)
                            temp[(6*ch+1)*ch_split:(6*ch+2)*ch_split,i,:] += np.random.normal(0,1,temp.shape[2])
                            temp[(6*ch+2)*ch_split:(6*ch+3)*ch_split,i,:] += np.random.normal(0,2,temp.shape[2])
                            temp[(6*ch+3)*ch_split:(6*ch+4)*ch_split,i,:] += np.random.normal(0,3,temp.shape[2])
                            temp[(6*ch+4)*ch_split:(6*ch+5)*ch_split,i,:] += np.random.normal(0,4,temp.shape[2])
                            temp[(6*ch+5)*ch_split:(6*ch+6)*ch_split,i,:] += np.random.normal(0,5,temp.shape[2])                    
                    elif noise_type == 'allmix':
                        if rep_i == 0:
                            temp[6*ch*ch_split:(6*ch+1)*ch_split,i,:] = 0
                            temp[(6*ch+2)*ch_split:(6*ch+3)*ch_split,i,:] += np.sin(2*np.pi*60*x)
                            temp[(6*ch+3)*ch_split:(6*ch+4)*ch_split,i,:] += 2*np.sin(2*np.pi*60*x)
                            temp[(6*ch+4)*ch_split:(6*ch+5)*ch_split,i,:] += 3*np.sin(2*np.pi*60*x) 
                            temp[(6*ch+5)*ch_split:(6*ch+6)*ch_split,i,:] += 4*np.sin(2*np.pi*60*x)
                            temp_split = temp[(6*ch+1)*ch_split:(6*ch+2)*ch_split,i,:]
                            for temp_iter in range(ch_split):
                                if ch_noise[temp_iter,ch_ind] == 0:
                                    temp_split[temp_iter,...] = 0
                                elif ch_noise[temp_iter,ch_ind] == 1:
                                    temp_split[temp_iter,...] += np.random.normal(0,ch_level[temp_iter,ch_ind]+1,temp.shape[2])
                                else:
                                    temp_split[temp_iter,...] += (ch_level[temp_iter,ch_ind]+1)*np.sin(2*np.pi*60*x)
                            temp[(6*ch+1)*ch_split:(6*ch+2)*ch_split,i,:] = cp.deepcopy(temp_split)
                        else:        
                            temp[(6*ch)*ch_split:(6*ch+1)*ch_split,i,:] += 5*np.sin(2*np.pi*60*x)
                            temp[(6*ch+1)*ch_split:(6*ch+2)*ch_split,i,:] += np.random.normal(0,1,temp.shape[2])
                            temp[(6*ch+2)*ch_split:(6*ch+3)*ch_split,i,:] += np.random.normal(0,2,temp.shape[2])
                            temp[(6*ch+3)*ch_split:(6*ch+4)*ch_split,i,:] += np.random.normal(0,3,temp.shape[2])
                            temp[(6*ch+4)*ch_split:(6*ch+5)*ch_split,i,:] += np.random.normal(0,4,temp.shape[2])
                            temp[(6*ch+5)*ch_split:(6*ch+6)*ch_split,i,:] += np.random.normal(0,5,temp.shape[2])
                    elif noise_type == 'testall':
                        if rep_i == 0:
                            for scale_i in range(5):
                                temp[(5*ch+scale_i)*ch_split:(5*ch+scale_i+1)*ch_split,i,:] += (scale_i+1)*np.sin(2*np.pi*60*x)
                        elif rep_i == 1:        
                            for scale_i in range(5):
                                temp[(5*ch+scale_i)*ch_split:(5*ch+scale_i+1)*ch_split,i,:] += np.random.normal(0,scale_i+1,temp.shape[2])
                        else:
                            temp[5*ch*ch_split:(5*ch+2)*ch_split,i,:] = 0
                            temp_split = temp[(5*ch+2)*ch_split:(5*ch+5)*ch_split,i,:]
                            for temp_iter in range(ch_split):
                                if ch_noise[temp_iter,ch_ind] == 0:
                                    temp_split[temp_iter,...] = 0
                                elif ch_noise[temp_iter,ch_ind] == 1:
                                    temp_split[temp_iter,...] += np.random.normal(0,ch_level[temp_iter,ch_ind]+1,temp.shape[2])
                                else:
                                    temp_split[temp_iter,...] += (ch_level[temp_iter,ch_ind]+1)*np.sin(2*np.pi*60*x)
                            temp[(5*ch+2)*ch_split:(5*ch+5)*ch_split,i,:] = cp.deepcopy(temp_split)
                    elif noise_type == 'flat':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] = 0
                    elif noise_type == 'gauss':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += np.random.normal(0,scale,temp.shape[2])
                    elif noise_type == '60hz':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += scale*np.sin(2*np.pi*60*x)
                    elif noise_type == 'mix':
                        temp_split = temp[ch*ch_split:(ch+1)*ch_split,i,:]
                        for temp_iter in range(ch_split):
                            if ch_noise[temp_iter,ch_ind] == 0:
                                temp_split[temp_iter,...] = 0
                            elif ch_noise[temp_iter,ch_ind] == 1:
                                temp_split[temp_iter,...] += np.random.normal(0,ch_level[temp_iter,ch_ind]+1,temp.shape[2])
                            else:
                                temp_split[temp_iter,...] += (ch_level[temp_iter,ch_ind]+1)*np.sin(2*np.pi*60*x)
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] = cp.deepcopy(temp_split)
                    elif noise_type == 'realcontact':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += real_noise[2,ch_noise[:,ch_ind],:] * emg_scale[i]
                    elif noise_type == 'realcontactbig':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += real_noise[3,ch_noise[:,ch_ind],:] * emg_scale[i]
                    elif noise_type == 'realbreak':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += real_noise[0,ch_noise[:,0],:] * emg_scale[i]
                    elif noise_type == 'realbreaknm':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += real_noise[1,ch_noise[:,0],:] * emg_scale[i]
                    elif noise_type == 'realmove':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += real_noise[-1,ch_noise[:,ch_ind],:] * emg_scale[i]
                    elif noise_type == 'realmixeven':
                        noise = real_noise[noise_all[:,ch_ind],ch_noise[:,ch_ind],:] * emg_scale[i]
                        noise[noise > 5] = 5
                        noise[noise < -5] = -5
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += noise
                    elif noise_type[:7] == 'realmix':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += real_noise[ch_level[:,ch_ind],ch_noise[:,ch_ind],:] * emg_scale[i]
                    
                    ch_ind += 1 

            out = np.concatenate((out,temp))
    
    out = np.concatenate((raw, out))

    noisy, clean, y = out, orig, to_categorical(sub_params[:,4]-1)

    clean = clean[...,np.newaxis]
    noisy = noisy[...,np.newaxis]

    noisy = truncate(noisy)
    clean = truncate(clean)
    return noisy,clean,y


def threshold(raw,params, z_mav=0):
    # raw format (samps x chan x win)
    if raw.shape[-1] == 1:
        raw = np.squeeze(raw)
    N = raw.shape[2]
    ch = raw.shape[1]
    th = 1.1

    mean_mav = np.tile(np.mean(raw,axis=2)[...,np.newaxis],(1,1,N))
    raw_demean = raw-mean_mav

    mav=np.sum(np.abs(raw_demean),axis=2)

    z_all = np.sum(mav[params[:,-1]==1,:], axis=1)
    if z_mav == 0:
        z_mav = th * np.mean(z_all,axis=0)
    mav_all = np.sum(mav, axis=1)
    ind = (params[:,-1]==1) | ((params[:,-1] != 1) & (mav_all > z_mav))
    raw_out = raw[ind,...]
    params_out = params[ind,...]
    raw_z = raw[~ind,...]
    params_z = np.tile([1,1,1],(raw_z.shape[0],1))
    raw_out = np.vstack((raw_z,raw_out))
    params_out = np.vstack((params_z,params_out))

    return raw_out, params_out, z_mav


def extract_feats_caps(raw,ft='feat',uint=False,order=6):
    # raw format (samps x chan x win)
    if raw.shape[-1] == 1:
        raw = np.squeeze(raw)
    N = raw.shape[2]
    ch = raw.shape[1]
    samp = raw.shape[0]
    th = 1.1

    if uint:
        z_th = 164
        s_th = 99
    else:
        z_th = 0.025
        s_th = 0.015

    mean_mav = np.tile(np.mean(raw,axis=2)[...,np.newaxis],(1,1,N))
    raw_demean = raw-mean_mav

    mav=np.sum(np.abs(raw_demean),axis=2)

    if ft != 'mav':
        last = raw_demean[...,:-2]
        next = raw_demean[...,1:]

        zero_change = (next*raw_demean[...,:-1] < 0) & ((np.abs(next) >= z_th) | (np.abs(raw_demean[...,:-1])>=z_th))
        zc = np.sum(zero_change, axis=2)

        next_s = next[...,1:] - raw_demean[...,1:-1]
        last_s = raw_demean[...,1:-1] - last
        sign_change = ((next_s > 0) & (last_s < 0)) | ((next_s < 0) & (last_s > 0))
        th_check = (np.abs(next_s) > s_th) | (np.abs(last_s) > (s_th))
        ssc = np.sum(sign_change & th_check, axis=2)

        wl = np.sum(np.abs(next - raw_demean[...,:-1]), axis=2)

        feat_out = np.concatenate([mav,wl,zc,ssc],-1)

        if ft == 'tdar':
            AR = np.zeros((samp,raw.shape[1],order))
            for ch in range(raw.shape[1]):
                AR[:,ch,:] = np.squeeze(matAR_ch(raw[:,ch,:],order))
            reg_out = np.real(AR.transpose(0,2,1)).reshape((samp,-1))
            feat_out = np.hstack([feat_out,reg_out])
    else:
        feat_out = mav
    feat_out = feat_out/200

    if not uint:
        feat_out[...,:ch*2] = (2**16-1)*feat_out[...,:ch*2]/10

    return feat_out


def extract_feats(raw,th=0.01,ft='feat',order=6,emg_scale=1):
    if raw.shape[-1] == 1:
        raw = np.squeeze(raw)
    N=raw.shape[2]
    samp = raw.shape[0]
    z_th = 0.025
    s_th = 0.015

    mav=np.sum(np.absolute(raw),axis=2)/N

    if ft != 'mav':
        last = np.roll(raw, 1, axis=2)
        next = np.roll(raw, -1, axis=2)

        # zero crossings
        zero_change = (next[...,:-1]*raw[...,:-1] < 0) & (np.absolute(next[...,:-1]-raw[...,:-1])>(emg_scale*z_th))
        zc = np.sum(zero_change, axis=2)

        # slope sign change
        next_s = next[...,1:-1] - raw[...,1:-1]
        last_s = raw[...,1:-1] - last[...,1:-1]
        sign_change = ((next_s > 0) & (last_s < 0)) | ((next_s < 0) & (last_s > 0))
        th_check = (np.absolute(next_s) >(emg_scale*s_th)) & (np.absolute(last_s) > (emg_scale*s_th))
        ssc = np.sum(sign_change & th_check, axis=2)

        # waveform length
        wl = np.sum(np.absolute(next[...,:-1] - raw[...,:-1]), axis=2)

        # feat_out = 0
        feat_out = np.concatenate([mav,wl,zc,ssc],-1)
        
        if ft == 'tdar':
            AR = np.zeros((samp,raw.shape[1],order))
            for ch in range(raw.shape[1]):
                AR[:,ch,:] = np.squeeze(matAR_ch(raw[:,ch,:],order))
            reg_out = np.real(AR.transpose(0,2,1)).reshape((samp,-1))
            feat_out = np.hstack([feat_out,reg_out])
    else:
        feat_out = mav
    return feat_out

def extract_scale(x,scaler=None,load=True, ft='feat',emg_scale=1,caps=False):
    # extract features 
    if ft == 'feat':
        num_feat = 4
    elif ft == 'tdar':
        num_feat = 10
    elif ft == 'mav':
        num_feat = 1
    
    if caps:
        x_temp = extract_feats_caps(x,ft=ft)
        x_temp = np.transpose(x_temp.reshape((x_temp.shape[0],num_feat,-1)),(0,2,1))[...,np.newaxis]
        x_test = x_temp.reshape(x_temp.shape[0]*x_temp.shape[1],-1)
        x_min = x_test.min(axis=0)
        x_max = x_test.max(axis=0)
        # X_std = (X - x_min)) / (x_max - x_min)
        # X_scaled = X_std * (max - min) + min
    else:
        x_temp = np.transpose(extract_feats(x,ft=ft,emg_scale=emg_scale).reshape((x.shape[0],num_feat,-1)),(0,2,1))[...,np.newaxis]
    
    # scale features
    if scaler is not None:
        if load:
            x_vae = scaler.transform(x_temp.reshape(x_temp.shape[0]*x_temp.shape[1],-1)).reshape(x_temp.shape)
        else:
            print('scaling')
            x_vae = scaler.fit_transform(x_temp.reshape(x_temp.shape[0]*x_temp.shape[1],-1)).reshape(x_temp.shape)
    else:
        x_vae = x_temp
    
    if caps:
        return x_vae, scaler, x_min, x_max
    else:
        return x_vae, scaler

def matAR_ch(data,order):
    datalen = data.shape[-1]
    num_ch = data.shape[0]
    AR = np.zeros((order+1,num_ch))
    K = np.zeros((order+1,num_ch))

    R0 = np.zeros((num_ch,))
    ix = 0
    iy = 0
    for k in range(datalen):
        R0 += np.multiply(data[:,ix],data[:,iy])
        ix += 1
        iy += 1

    R = np.zeros((order,num_ch))

    for i in range(order):
        if 1 > (datalen - (1 + i)):
            i0 = 0
        else:
            i0 = datalen - (1 + i)

        if ((1 + i) + 1) > datalen:
            i1 = 1
        else:
            i1 = (1 + i) + 1
        
        q = np.zeros((num_ch,))
        if i0 >= 1:
            ix = 0
            iy = 0
            for k in range(1,i0+1):
                q += np.multiply(data[:,ix],data[:,(i1 + iy)-1])
                ix += 1
                iy += 1
        
        R[i,:] = q

    AR[1,:] = np.divide(-R[0,:],R0)
    K[0,:] = AR[1,:]
    q = np.full((num_ch,),R[0,:])
    temp = np.zeros((order,num_ch))

    for i in range(order-1):
        R0 += np.multiply(q,K[i,:])
        q = np.zeros((num_ch,))
        for k in range(i+1):
            b = AR[(((1+i)+2) - (1 + k)) - 1,:]
            q += np.multiply(R[k,:], b)

        q += R[((1+i)+1)-1,:]
        K[1+i,:] = np.divide(-q,R0)
        for k in range(i+1):
            b = AR[(((1+i)+2)-(1+k)) - 1,:]
            temp[k,:] = np.multiply(K[((1+i)+1)-1,:],b)
        
        for k in range(((1+i)+1) - 1):
            AR[k+1,:] += temp[k,:]
        
        AR[(1+i)+1,:] = K[1+i,:]
    
    AR = np.nan_to_num(AR).T
    return AR[:,1:]