#!/usr/bin/python
# Put the scripts into a subfolder of the AVEC2018_CES package, e.g., AVEC2018_CES/scripts_CES/
# Output: csv files with functionals of low-level descriptors (LLDs)

import os
import fnmatch
import numpy as np
from read_csv import load_features
from write_csv import save_features

# Folders with feature files
folder_lld_features = ['../audio_features_mfcc/',
                       '../audio_features_egemaps/',
                       '../visual_features/']

folder_functionals  = ['../audio_features_mfcc_functionals/',
                       '../audio_features_egemaps_functionals/',
                       '../visual_features_functionals/']

# frames per second for each feature type in folder_lld_features
fps = [100, 100, 50] # TODO: add correct number of frames per second for the 3rd item (corresponding to video), 100 is correct for mfcc and egemaps, but video frame right might be lower than 50 Hz -> check and adapt!

# Window size
window_size = 4.0  # seconds

# Do NOT modify
hop_size    = 0.1  # hop size of the labels TODO
max_seq_len = 1768 # TODO, this should at least the maximum number of labels per file (=duration / hop_size), but can be larger as files are cropped later on (just used for allocation)

# Get all files
files = fnmatch.filter(os.listdir(folder_lld_features[0]), '*.csv')  # filenames are the same for all modalities
files.sort()

# Generate files with functionals for all modalities and train/devel/test
for m in range(0, len(folder_lld_features)):
    if not os.path.exists(folder_functionals[m]):
        os.mkdir(folder_functionals[m])
    
    for fn in files:
        X = load_features(folder_lld_features[m] + fn, skip_header=True, skip_instname=True, delim=';')
        print(fn)
        print(X.shape)
        num_llds = X.shape[1]-1  # (time stamp is not considered as feature)
        X_func = np.zeros(( max_seq_len, num_llds*2 ))  # mean + stddev for each LLD
        window_size_half = int(window_size * fps[m] / 2)
        
        time_stamps_new = np.empty((max_seq_len,1))
        for t in range(0, max_seq_len):
            t_orig   = int(t * fps[m] * hop_size)
            min_orig = max(0, t_orig-window_size_half)
            max_orig = min(X.shape[0], t_orig+window_size_half+1)
            if min_orig<max_orig and t_orig<=X.shape[0]:  # X can be smaller, do not consider 
                time_stamps_new[t] = t * hop_size
                X_func[t, :num_llds] = np.mean(X[min_orig:max_orig,1:], axis=0)  # skip time stamp
                X_func[t, num_llds:] = np.std(X[min_orig:max_orig,1:],  axis=0)  # skip time stamp
            else:
                time_stamps_new = time_stamps_new[:t,:]
                X_func          = X_func[:t,:]
                break        
        X_func = np.concatenate((time_stamps_new, X_func), axis=1)
        
        save_features(folder_functionals[m] + fn, X_func, append=False, instname=fn[:-4], header='', delim=';', precision=6, first_time=True)


