import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def create_dataframe(acc_clean,acc_noise):
    acc_clean[...,-1] = acc_clean[...,10]
    acc_clean = acc_clean[~np.isnan(acc_clean[:,0,0,0]),...]
    acc_noise = acc_noise[~np.isnan(acc_noise[:,0,0,0]),...]

    data = np.squeeze(np.hstack((acc_clean[:,[0],:,:], acc_noise))).reshape([-1])
    mask = np.ones((data.shape))
    mask[np.isnan(data)] = 0
    data = data[mask.astype(bool)]

    sub = 1
    sub_array = np.zeros(data.shape)
    temp_elec = np.zeros((acc_clean.shape[-1]*5,))
    elec = 0
    for i in range(0,acc_clean.shape[-1]*5,acc_clean.shape[-1]):
        temp_elec[i:i+acc_clean.shape[-1]] = elec
        elec+=1
    elec_array = np.tile(temp_elec,(acc_clean.shape[0],))
    for i in range(0,data.shape[0],acc_clean.shape[-1]*5):
        sub_array[i:i+acc_clean.shape[-1]*5,] = sub
        sub+=1
    mod_array = np.tile(np.arange(acc_clean.shape[-1]),(acc_clean.shape[0]*5,))
    data = 100*(1-data)
    data = np.stack((data,sub_array,elec_array,mod_array))
    df = pd.DataFrame(data.T,columns=['acc','sub','elec','mod'])

    df['elec2'] = df['elec']**2
    df['elec3'] = df['elec']**3
    df['elec4'] = df['elec']**4

    return df

def get_mods(df):
    if 0: # old models
        out_df = df[(df['mod']!=0) & (df['mod']!=3)& (df['mod']!=5)& (df['mod']!=8)& (df['mod']!=12)& (df['mod']!=13)]
    else:
        out_df = df[(df['mod']==7) | (df['mod']==14)| (df['mod']==11)| (df['mod']==10)| (df['mod']==6) | (df['mod']==5) | (df['mod']==4)]

    return out_df

def run_fit(df,ctrl=10):
    md = smf.mixedlm("acc ~ C(mod,Treatment(" + str(ctrl) + ")) + C(mod,Treatment(" + str(ctrl) + "))*elec", df,groups=df["sub"])

    all_md = {}

    mdf = md.fit()
    print(mdf.summary())
    all_md['main'] = mdf

    for i in np.unique(df['elec']):
        print(i)
        new_df = df[df['elec'] == i]
        md2 = smf.mixedlm("acc ~ C(mod,Treatment(" + str(ctrl) + "))", new_df, groups=new_df["sub"])
        mdf2 = md2.fit()
        print(mdf2.summary())
        all_md[str(i)] = mdf2

    return all_md