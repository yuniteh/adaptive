from re import A
import numpy as np
import scipy.io 
import copy as cp
import pickle
import os
from datetime import date
from tensorflow.keras.utils import to_categorical
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

def load_noise_data(filename):
    struct = scipy.io.loadmat(filename)
    raw_noise = struct['raw_win']

    return raw_noise
    
def truncate(data):
    data[data > 5] = 5
    data[data < -5] = -5
    return data

def prep_train_caps(x_train, params, prop_b=True, num_classes=None, batch_size=128,noise=True, scaler=None, emg_scale=None, ft='feat', split=False):
    if split:
        x_rest = x_train[params[:,-1] == 1,...]
        x_train_half = x_train[(params[:,1]%2==0) & (params[:,-1] != 1),...]
        x_train = np.vstack((x_rest[:x_rest.shape[0]//2,...],x_train_half))

        p_rest = params[params[:,-1] == 1,...]
        p_train_half = params[(params[:,1]%2==0) & (params[:,-1] != 1),...]
        params = np.vstack((p_rest[:p_rest.shape[0]//2,...],p_train_half))

    x_train, params = shuffle(x_train, params, random_state = 0)
    x_orig = cp.deepcopy(x_train)
    
    emg_scale = np.ones((x_train.shape[1],1))
    if not isinstance(emg_scale,np.ndarray):
        emg_scale = np.ones((np.size(x_train,1),1))
        for i in range(np.size(x_train,1)):
            emg_scale[i] = 5/np.max(np.abs(x_train[:,i,:]))
    x_train *= emg_scale

    y = to_categorical(params[:,0],num_classes=num_classes)
    x_train_clean, y_train_clean = shuffle(x_train,y,random_state=0)

    if noise:
        x_train_noise, _, y_train_noise = add_noise_caps(x_train, params, num_classes=num_classes)
        # shuffle data to make even batches
        x_train_noise, y_train_noise = shuffle(x_train_noise, y_train_noise, random_state = 0)
    else:
        x_train_noise = cp.deepcopy(x_train_clean)
        y_train_noise = cp.deepcopy(y_train_clean)

    # Extract features
    if scaler is not None:
        load = True
    else:
        scaler = MinMaxScaler(feature_range=(0,1))
        load = False
        
    x_train_noise_cnn, scaler, x_min, x_max= extract_scale(x_train_noise,scaler=scaler,load=load,ft=ft,caps=True) 
    x_train_noise_cnn = x_train_noise_cnn.astype('float32')        

    # reshape data for nonconvolutional network
    x_train_noise_mlp = x_train_noise_cnn.reshape(x_train_noise_cnn.shape[0],-1)

    # clean data
    x_train_clean_cnn, _, _, _= extract_scale(x_train_clean,scaler=scaler,load=True,ft=ft,caps=True) 
    x_train_clean_cnn = x_train_clean_cnn.astype('float32')        

    # reshape data for nonconvolutional network
    x_train_clean_mlp = x_train_clean_cnn.reshape(x_train_clean_cnn.shape[0],-1)

    # LDA data
    y_train_lda = params[:,[0]]
    x_train_lda = extract_feats_caps(x_orig,ft=ft)
    # y_train_lda = np.argmax(y_train_noise,axis=1)[...,np.newaxis]
    # x_train_lda = extract_feats_caps(x_train_noise,ft=ft)

    return x_train_clean_mlp, x_train_clean_cnn, y_train_clean, x_train_noise_mlp, x_train_noise_cnn, y_train_noise, x_train_lda, y_train_lda, emg_scale, scaler, x_min, x_max

def prep_test_caps(x, params, scaler=None, emg_scale=None, num_classes=None,ft='feat', split=False):
    if split:
        x_rest = x[params[:,2] == 1,...]
        x_test_half = x[(params[:,1]%2!=0) & (params[:,2] !=1),...]
        x = np.vstack((x_rest[:x_rest.shape[0]//2,...],x_test_half))

        p_rest = params[params[:,2] == 1,...]
        p_test_half = params[(params[:,1]%2!=0) & (params[:,2] !=1),...]
        params = np.vstack((p_rest[:p_rest.shape[0]//2,...],p_test_half))

    x, params = shuffle(x, params, random_state = 0)
    y = to_categorical(params[:,0],num_classes=num_classes)

    x_orig = cp.deepcopy(x)
    emg_scale = np.ones((x.shape[1],1))
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
    y_lda = params[:,[0]]
    x_lda = extract_feats_caps(x_orig,ft=ft)
    # y_lda = np.argmax(y_train_noise,axis=1)
    # x_lda = extract_feats_caps(x_train_noise,ft=ft)

    return y_test, x_test_mlp, x_test_cnn, x_lda, y_lda

def add_noise_caps(raw, params,num_classes=None):
    all_ch = raw.shape[1]
    num_ch = 5
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
                        # print(np.random.normal(0,.001,temp.shape[2]).shape)
                        temp[6*ch*ch_split:(6*ch+1)*ch_split,i,:] = np.random.normal(0,.001,temp.shape[2]) #0
                        temp[(6*ch+2)*ch_split:(6*ch+3)*ch_split,i,:] += np.sin(2*np.pi*60*x)
                        temp[(6*ch+3)*ch_split:(6*ch+4)*ch_split,i,:] += 2*np.sin(2*np.pi*60*x)
                        temp[(6*ch+4)*ch_split:(6*ch+5)*ch_split,i,:] += 3*np.sin(2*np.pi*60*x) 
                        temp[(6*ch+5)*ch_split:(6*ch+6)*ch_split,i,:] += 4*np.sin(2*np.pi*60*x)
                        temp_split = temp[(6*ch+1)*ch_split:(6*ch+2)*ch_split,i,:]
                        for temp_iter in range(ch_split):
                            if ch_noise[temp_iter,ch_ind] == 0:
                                temp_split[temp_iter,...] = np.random.normal(0,.001,temp.shape[2]) #0
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

    noisy, clean, y = out, orig, to_categorical(sub_params[:,0],num_classes=num_classes)

    clean = clean[...,np.newaxis]
    noisy = noisy[...,np.newaxis]

    noisy = truncate(noisy)
    clean = truncate(clean)
    return noisy,clean,y

def aug_gen(raw, params, mod, scaler, out_scaler):
    all_ch = raw.shape[1]
    num_ch = 5
    split = 6
    split = 12
    rep = 2
    start_ch = 1

    out = np.array([]).reshape(0,all_ch,raw.shape[2],1)
    m = list(params.shape[1:])
    m.insert(0,0)
    y_out = np.array([]).reshape(m)

    # loop through channel noise
    for num_noise in range(start_ch,num_ch):
        # find all combinations of noisy channels
        ch_all = np.array(list(combinations(range(0,all_ch),num_noise)))
        ch_tile = np.ones((ch_all.ndim,)).astype(int)
        ch_tile[0] = int((raw.shape[0]*rep)//ch_all.shape[0])
        # temp = np.tile(raw[:ch_tile[0]*ch_all.shape[0],...],(rep,1,1,1))
        temp = np.tile(raw,(rep,1,1,1))[:int(ch_tile[0]*ch_all.shape[0]),...]
        y = np.tile(params,(rep,1))[:int(ch_tile[0]*ch_all.shape[0]),...]

        ch_split = int(temp.shape[0]//split)
        ch_tot = np.tile(ch_all,ch_tile)
        temp = temp[:ch_split*split,...]
        y = y[:ch_split*split,...]
        ch_tot = ch_tot[:ch_split*split,...]
        temp_full = np.zeros((temp.shape[0],))
        ch_rand = np.random.randint(11,size = (ch_split,num_noise))

        temp_full[:ch_split] = 0
        for amp in range(1,6):
            temp_full[amp*ch_split:(amp+1)*ch_split] = amp/10
            temp_full[(amp+5)*ch_split:(amp+6)*ch_split] = (amp+5)/10

        for i in range(num_noise):
            temp_full[-ch_split:] = ch_rand[:,i]/10
            temp[np.arange(temp.shape[0]),ch_tot[:,i],:] = out_scaler.inverse_transform(mod(scaler.transform(np.squeeze(temp[np.arange(temp.shape[0]),ch_tot[:,i],:])),temp_full))[...,np.newaxis]
        
        y_out = np.concatenate((y_out,y))
        out = np.concatenate((out,temp))
    
    out = np.concatenate((raw, out))
    y_out = np.concatenate((params,y_out))

    # noisy, clean, y = out, orig, sub_params
    
    return out,0,y_out,temp_full

def add_noise_aug(raw,ft='tdar'):
    all_ch = raw.shape[1]
    split = 11

    out = np.array([]).reshape(0,all_ch,200)
    x = np.linspace(0,0.2,200)

    # loop through channel noise
    ch_all = np.arange(all_ch)
    ch_split = raw.shape[0]//(split)
    raw = raw[:ch_split*split,...]
    temp = cp.deepcopy(raw)
    t_label = np.zeros((temp.shape[0],all_ch))

    temp[:ch_split,:,:] = np.random.normal(0,.001,temp.shape[2]) #0
    t_label[:ch_split,:] = 0
    for amp in range(1,6):
        sig_60 = amp*np.sin(2*np.pi*60*x)
        temp[amp*ch_split:(amp+1)*ch_split,:,:] += sig_60
        t_label[amp*ch_split:(amp+1)*ch_split,:] = amp/10
        sig_norm = np.random.normal(0,amp,temp.shape[2])
        temp[(amp+5)*ch_split:(amp+6)*ch_split,:,:] += sig_norm
        t_label[(amp+5)*ch_split:(amp+6)*ch_split,:] = (amp+5)/10

    out = np.concatenate((out,temp))
    
    noisy, clean = out, raw

    clean = clean[...,np.newaxis]
    noisy = noisy[...,np.newaxis]

    noisy = truncate(noisy)
    clean = truncate(clean)

    x_clean, _, _, _= extract_scale(clean,scaler=None,ft=ft,caps=True) 
    x_noise, _, _, _= extract_scale(noisy,scaler=None,ft=ft,caps=True) 

    x_clean = x_clean.reshape((x_clean.shape[0]*x_clean.shape[1],x_clean.shape[2]))
    x_noise = x_noise.reshape((x_noise.shape[0]*x_noise.shape[1],x_noise.shape[2]))
    t_label = t_label.reshape((t_label.shape[0]*t_label.shape[1],))

    scaler = MinMaxScaler(feature_range=(-1,1))
    x_scaled = scaler.fit_transform(x_clean)
    out_scaler = MinMaxScaler(feature_range=(-1,1))
    x_n_scaled = out_scaler.fit_transform(x_noise)
    
    return t_label,x_clean,x_noise,x_scaled,x_n_scaled,scaler,out_scaler

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
        feat_out = feat_out/200
        # if not uint:
        #     feat_out[...,:ch*2] = (2**16-1)*feat_out[...,:ch*2]/10

        if ft == 'tdar':
            AR = np.zeros((samp,raw.shape[1],order))
            for ch in range(raw.shape[1]):
                # print(matAR(raw[:,ch,:],order).shape)
                AR[:,ch,:] = np.squeeze(matAR(raw[:,ch,:],order))
                # AR[:,ch,:] = np.squeeze(matAR_ch(raw[:,ch,:],order))
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
    else:
        x_temp = np.transpose(extract_feats(x,ft=ft,emg_scale=emg_scale).reshape((x.shape[0],num_feat,-1)),(0,2,1))[...,np.newaxis]
    
    # scale features
    if scaler is not None:
        if load:
            x_vae = scaler.transform(x_temp.reshape(x_temp.shape[0]*x_temp.shape[1],-1)).reshape(x_temp.shape)
        else:
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

def matAR1(data,order):
    data = data.astype('float32')*10/(2**16-1)-5
    AR = np.zeros((order+1,1))
    K = np.zeros((order+1,1))
    AR[0,0] = 1
    R0 = np.dot(data,data)
    R = np.zeros((1,order))
    for i in range(order):
        R[0,i] = np.dot(data[:-1*(i+1)],data[i+1:])
    E = cp.deepcopy(R0)
    AR[1,0] = -R[0,0]/R0
    K[0,0] = AR[1,0]
    q = R[0,0]
    tmp = np.zeros((1,order))

    for i in range(order-1):
        E += q*K[i,0]
        q = R[0,i+1]
        S = 0
        for k in range(i+1):
            S += R[0,k] * AR[i+1-k,0]

        q += S
        K[i+1,0] = -q/E
        for k in range(i+1):
            tmp[0,k] = K[i+1,0] * AR[i+1-k,0]

        for k in range(1,i+2):
            AR[k,0] = AR[k,0] + tmp[0,k-1]

        AR[i+2,0] = K[i+1,0]

    return AR[1:,0]

def matAR(data,order):
    samp = data.shape[0]
    # data = data.astype('float32')*10/(2**16-1)-5
    AR = np.zeros((order+1,samp))
    K = np.zeros((order+1,samp))
    AR[0,:] = 1
    R0 = np.sum(np.multiply(data,data),axis=1)
    R = np.zeros((samp,order))
    for i in range(order):
        R[:,i] = np.sum(np.multiply(data[:,:-1*(i+1)],data[:,i+1:]),axis=1)
    E = cp.deepcopy(R0)
    AR[1,:] = -R[:,0]/R0
    K[0,:] = AR[1,:]
    q = cp.deepcopy(R[:,0])
    tmp = np.zeros((samp,order))

    for i in range(order-1):
        E += np.multiply(q,K[i,:].T)
        q = cp.deepcopy(R[:,i+1])
        S = np.zeros((samp,))
        for k in range(i+1):
            S[:] += np.multiply(R[:,k],AR[i+1-k,:].T)

        q += S
        K[i+1,:] = -q/E
        for k in range(i+1):
            tmp[:,k] = np.multiply(K[i+1,:],AR[i+1-k,:])

        for k in range(1,i+2):
            AR[k,:] = AR[k,:] + tmp[:,k-1]

        AR[i+2,:] = K[i+1,:]

    return AR[1:,:].T

def set_mean(data,label, key, N=0,mu_class=None,std_class=None):
    if not isinstance(data,np.ndarray):
        data = data.numpy()
    m = list(data.shape[1:])
    u_class = np.unique(np.argmax(label,axis=1))
    n_class = u_class.shape[0]
    m.insert(0,n_class)
    if mu_class is None:
        mu_class = np.zeros(m)
        std_class = np.zeros(m)
        N = np.zeros([n_class,])

    ALPHA = np.zeros(N.shape)
    N_new = np.zeros((n_class,))
    
    for i in key:
        ind = np.squeeze(np.argmax(label,axis=1) == u_class[i])
        N_new[i] = np.sum(ind)
        ALPHA[i] = N[i] / (N[i] + N_new[i])
        mu_class[i,...] = ALPHA[i] * mu_class[i,...] + (1 - ALPHA[i]) * np.mean(data[ind,...],axis=0)                       # Update the mean vector
        std_class[i,...] = ALPHA[i] * std_class[i,...] + (1 - ALPHA[i]) * np.std(data[ind,...],axis=0)
        N[i] += N_new[i]

    return mu_class, std_class, N

def update_mean(data,label, N=0,mu_class=None,std_class=None,key=None,prev_key=None):
    if not isinstance(data,np.ndarray):
        data = data.numpy()
    m = list(data.shape[1:])

    key = key.astype(int)
    prev_key = prev_key.astype(int)

    m.insert(0,len(key))
    N_new = np.zeros((len(key,)))
    N_fixed = np.zeros((len(key,)))
    mu_fixed = np.zeros(m)
    std_fixed = np.zeros(m)

    for k in prev_key:
        N_fixed[k] = N[k]
        mu_fixed[k,...] = mu_class[k,...]
        std_fixed[k,...] = std_class[k,...]
    N = cp.deepcopy(N_fixed)
    mu_class = cp.deepcopy(mu_fixed)
    std_class = cp.deepcopy(std_fixed)
    ALPHA = np.zeros(N.shape)
    
    # for i in range(len(prev_key)):
    old_class = np.isin(key,prev_key,assume_unique=True)
    for i in key[old_class]:
        # ind = np.squeeze(np.argmax(label,axis=1) == i)
        ind = np.squeeze(label[:,i] == 1)
        N_new[i] = np.sum(ind)
        if N_new[i] > 0:
            ALPHA[i] = N[i] / (N[i] + N_new[i])
            mu_class[i,...] = ALPHA[i] * mu_class[i,...] + (1 - ALPHA[i]) * np.mean(data[ind,...],axis=0)                       # Update the mean vector
            std_class[i,...] = ALPHA[i] * std_class[i,...] + (1 - ALPHA[i]) * np.std(data[ind,...],axis=0)
            N[i] += N_new[i]
    
    new_class = ~np.isin(key,prev_key, assume_unique=True)
    for i in key[new_class]:
        print('new class avcnn')
        # ind = np.squeeze(np.argmax(label,axis=1) == i)
        ind = np.squeeze(label[:,i] == 1)
        N_new[i] = np.sum(ind)
        mu_class[i,...] = np.mean(data[ind,...],axis=0,keepdims=True)
        std_class[i,...] = np.std(data[ind,...],axis=0)
        
        N[i] += N_new[i]

    return mu_class, std_class, N

def update_mean_old(data,label, N=0,mu_class=None,std_class=None,key=None,prev_key=None):
    if not isinstance(data,np.ndarray):
        data = data.numpy()
    m = list(data.shape[1:])

    key = key.astype(int)
    prev_key = prev_key.astype(int)

    m.insert(0,len(key))
    N_new = np.zeros((len(key,)))
    N_fixed = np.zeros((len(key,)))
    mu_fixed = np.zeros(m)
    std_fixed = np.zeros(m)

    for k in prev_key:
        N_fixed[key == k] = N[prev_key == k]
        mu_fixed[key == k,...] = mu_class[prev_key==k,...]
        std_fixed[key == k,...] = std_class[prev_key==k,...]
    N = cp.deepcopy(N_fixed)
    mu_class = cp.deepcopy(mu_fixed)
    std_class = cp.deepcopy(std_fixed)
    ALPHA = np.zeros(N.shape)
    
    # for i in range(len(prev_key)):
    old_class = np.isin(key,prev_key,assume_unique=True)
    for k in key[old_class]:
        ind = np.squeeze(np.argmax(label,axis=1) == k)
        i = np.squeeze(key == k)
        N_new[i] = np.sum(ind)
        if N_new[i] > 0:
            ALPHA[i] = N[i] / (N[i] + N_new[i])
            mu_class[i,...] = ALPHA[i] * mu_class[i,...] + (1 - ALPHA[i]) * np.mean(data[ind,...],axis=0)                       # Update the mean vector
            std_class[i,...] = ALPHA[i] * std_class[i,...] + (1 - ALPHA[i]) * np.std(data[ind,...],axis=0)
            N[i] += N_new[i]
    
    new_class = ~np.isin(key,prev_key, assume_unique=True)
    for k in key[new_class]:
        print('new class avcnn')
        ind = np.squeeze(np.argmax(label,axis=1) == k)
        i = np.squeeze(key==k)
        N_new[i] = np.sum(ind)
        mu_class[i,...] = np.mean(data[ind,...],axis=0,keepdims=True)
        std_class[i,...] = np.std(data[ind,...],axis=0)
        
        N[i] += N_new[i]

    return mu_class, std_class, N

def hellinger(m1,s1,m2,s2):
    temp = ((np.linalg.det(s1)**.25) * np.linalg.det(s2)**.25)/(np.linalg.det((s1+s2)/2)**.5)
    if (np.linalg.det(s1 + s2)/2) != 0:
        covXY_inverted = np.linalg.inv((s1 + s2)/2)
    else:
        covXY_inverted = np.linalg.pinv((s1 + s2)/2)
    H = 1 - temp * np.exp((-1/8)*np.dot(np.dot((m1-m2).T,covXY_inverted),(m1-m2)))       

    meanX = m1
    meanY = m2
    covX = s1
    covY = s2
    detX = np.linalg.det(covX)
    detY = np.linalg.det(covY)
    detXY = np.linalg.det((covX + covY)/2)
    if (np.linalg.det(covX + covY)/2) != 0:
        covXY_inverted = np.linalg.inv((covX + covY)/2)
    else:
        covXY_inverted = np.linalg.pinv((covX + covY)/2)    
    dist = 1 - ((detX**.25 * detY**.25) / detXY**.5) * np.exp(-.125 * np.dot(np.dot(np.transpose(meanX-meanY),covXY_inverted),(meanX - meanY)))
    # print(temp)
    return np.sum(np.sum(dist.astype(np.longdouble)))

def hellinger3(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

def bhatta(m1,s1,m2,s2):
    print(np.linalg.det(s2))
    b = .125 * (m1-m2).T*np.linalg.inv((s1+s2)/2)*(m1-m2)+.5*np.log((np.linalg.det((s1+s2)/2))/(np.sqrt(np.linalg.det(s2)*np.linalg.det(s1))))
    return np.nanmean(np.nanmean(b))

def mahal(m1,s1,m2,s2):
    if (np.linalg.det((s1+s2)/2)) != 0:
        cov_inv = np.linalg.inv((s1+s2)/2)
    else:
        cov_inv = np.linalg.pinv((s1+s2)/2)
    m = np.sqrt((m1-m2).T*(cov_inv)*(m1-m2))
    return m

def split_data(train_data,train_params):
    tr_i = np.zeros((train_params.shape[0],))
    te_i = np.zeros((train_params.shape[0],))
    for cls in np.unique(train_params[:,-1]):
        dof = np.array(np.where(train_params[:,-1] == cls))
        tr_i[dof[0,:dof.shape[1]//2]] = 1
        te_i[dof[0,dof.shape[1]//2:]] = 1

    train_temp = train_data[tr_i.astype(bool),...]
    params_temp = train_params[tr_i.astype(bool),...]
    val_data = train_data[te_i.astype(bool),...]
    val_params = train_params[te_i.astype(bool),...]

    train_data, train_params = train_temp, params_temp
    return train_data, train_params, val_data, val_params