########################################################################################################################
# FILE            : adaptiveVR_VirtualCoach.py
# VERSION         : 9.0.0
# FUNCTION        : Adapt upper limb data using an LDA classifier. To be used with Unity project.
# DEPENDENCIES    : None
# SLAVE STEP      : Replace Classify
#_author__ = 'lhargrove & rwoodward & yteh'
########################################################################################################################

# Import all the required modules. These are helper functions that will allow us to get variables from CAPS PC
import os

from numpy import extract
import pcepy.pce as pce
import pcepy.feat as feat
import numpy as np
import copy as cp
import time

# Class dictionary
classmap = [1,10,11,12,13,16,19]
# Specify where the saved data is stored.
datafolder = 'DATA'
datadir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', datafolder))
# Number of modes/classes.
numModes = int(len(classmap))
# Number of EMG channels.
numEMG = int(len(pce.get_var('DAQ_CHAN').to_np_array()[0]))
# Feature value ('47'; time domain and autoregression)
featVal = 15
# Number of features. (10 for '47')
featNum = 4
# Matrix size.
matSize = numEMG * featNum
# Threshold multiplier
thresX = 1.1
# Sample threshold
samp_thres = 100
# Voltage range of EMG signal (typically +/- 5V)
voltRange = 5
# True: enhanced proportional control is used, otherwise incumbent.
useEnhanced = True
# True: use CAPS MAV method, otherwise use self-calculated method.
CAPSMAV = False
# True: ramp enabled, otherwise ramp disabled.
rampEnabled = True
# Ramp time (in ms)
rampTime = 500
# Define the starting ramp numerators and denominators.
ramp_numerator = np.zeros((1, numModes), dtype=float, order='F')
ramp_denominator = np.ones((1, numModes), dtype=float, order='F') * (rampTime / pce.get_var('DAQ_FRINC'))
# DAQ UINT ZERO
DAQ_conv = (2**16-1)/2

def dispose():
    pass

############################################# MAIN FUNCTION LOOP #######################################################
def run():
    # Don't do anything if PCE is training.
    # Get raw DAQ data for the .
    raw_DAQ = np.array(pce.get_var('DAQ_DATA').to_np_array()[0:numEMG,:], order='F')
    scaled_raw = (raw_DAQ.astype('float') - DAQ_conv) + DAQ_conv ## might be problem
    feat_scaled = feat.extract(featVal, scaled_raw.astype('uint16')) ## size = 1x24 (numfeat)
    pce.set_var('FEAT_SCALED', feat_scaled.astype(float, order='F'))
                