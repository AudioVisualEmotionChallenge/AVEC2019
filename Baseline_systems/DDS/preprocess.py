import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.io import loadmat
import os

def convert_file_to_pickle(path, feature_set):
    
    if "OpenFace" in feature_set:
        df = pd.read_csv(path, header=0)
    elif path.endswith('.csv'):
        df = pd.read_csv(path, header=0, sep=';')
    elif path.endswith('.mat'):
        df = pd.DataFrame( loadmat(path)['feature'])

    col_ind = 4 if "OpenFace" in feature_set else 0 if '.mat' in feature_set else 2

    if not path.endswith('.mat'):
        out_df = df.iloc[:,col_ind:]
        return out_df
    else: 
        out_df = df.iloc[:,col_ind:]
        return out_df.values

def subsample_frames(file_np):

    mask = np.ones(file_np.shape[0], dtype=bool)
    mask[np.arange(1, file_np.shape[0], 2)] = False
    file_np = file_np[mask]
    return file_np

def smooth_subsample_frames(file_np):

    smoothed_np = savgol_filter(file_np, window_length=11, polyorder=5, axis=0)[0::4]
    return smoothed_np

def preprocess1(path, feature_set, split_path, out_path):

    subjects = os.listdir(path)
    subjects.sort()
    scaler = StandardScaler()

    for subj in subjects:
        subj_p = os.path.join(path, subj, 'features', subj[:-1]+feature_set)
        file_np = convert_file_to_pickle(subj_p, feature_set)
        file_np = scaler.fit_transform(file_np)
        np.save(os.path.join(out_path, subj[:-2])+'.npy', file_np)

def preprocess2(path, feature_set, split_path, out_path):

    train = pd.read_csv(os.path.join(split_path, 'train_split.csv'), header=0)['Participant_ID'].tolist()
    dev = pd.read_csv(os.path.join(split_path, 'dev_split.csv'), header=0)['Participant_ID'].tolist()
    test = pd.read_csv(os.path.join(split_path, 'test_split.csv'), header=0)['Participant_ID'].tolist()

    for i in range(1, len(train)+1):
        fname = 'training_' + str(i).zfill(3) + "_" + feature_set
        file_np = pd.read_csv(os.path.join(path, fname), header=None, sep=';').iloc[:,2:].values
        np.save(os.path.join(out_path, str(train[i-1]))+'.npy', file_np)
    for i in range(1, len(dev)+1):
        fname = 'development_' + str(i).zfill(2) + "_" + feature_set
        file_np = pd.read_csv(os.path.join(path, fname), header=None, sep=';').iloc[:,2:].values
        np.save(os.path.join(out_path, str(dev[i-1]))+'.npy', file_np)
    for i in range(1, len(dev)+1):
        fname = 'test_' + str(i).zfill(2) + "_" + feature_set
        file_np = pd.read_csv(os.path.join(path, fname), header=None, sep=';').iloc[:,2:].values
        np.save(os.path.join(out_path, str(test[i-1]))+'.npy', file_np)

def preprocess3(path, feature_set, split_path, out_path):

    subjects = os.listdir(path)
    subjects.sort()

    for subj in subjects:
        subj_p = os.path.join(path, subj, 'features', subj[:-1]+feature_set)
        file_np = convert_file_to_pickle(subj_p, feature_set)
        file_np = subsample_frames(file_np)
        np.save(os.path.join(out_path, subj[:-2])+'.npy', file_np)

def preprocess4(path, feature_set, split_path, out_path):

    subjects = os.listdir(path)
    subjects.sort()

    for subj in subjects:
        subj_p = os.path.join(path, subj, 'features', subj[:-1]+feature_set)
        file_np = convert_file_to_pickle(subj_p, feature_set)
        file_np = smooth_subsample_frames(file_np)
        np.save(os.path.join(out_path, subj[:-2])+'.npy', file_np)

def preprocess5(path, feature_set, split_path, out_path):

    train = pd.read_csv(os.path.join(split_path, 'train_split.csv'), header=0)['Participant_ID'].tolist()
    dev = pd.read_csv(os.path.join(split_path, 'dev_split.csv'), header=0)['Participant_ID'].tolist()
    test = pd.read_csv(os.path.join(split_path, 'test_split.csv'), header=0)['Participant_ID'].tolist()

    for i in range(1, len(train)+1):
        fname = 'training_' + str(i).zfill(3) + "_" + feature_set
        file_np = pd.read_csv(os.path.join(path, fname), header=None, sep=';').iloc[:,2:].values
        file_np = subsample_frames(file_np)
        np.save(os.path.join(out_path, str(train[i-1]))+'.npy', file_np)
    for i in range(1, len(dev)+1):
        fname = 'development_' + str(i).zfill(2) + "_" + feature_set
        file_np = pd.read_csv(os.path.join(path, fname), header=None, sep=';').iloc[:,2:].values
        file_np = subsample_frames(file_np)
        np.save(os.path.join(out_path, str(dev[i-1]))+'.npy', file_np)
    for i in range(1, len(dev)+1):
        fname = 'test_' + str(i).zfill(2) + "_" + feature_set
        file_np = pd.read_csv(os.path.join(path, fname), header=None, sep=';').iloc[:,2:].values
        file_np = subsample_frames(file_np)
        np.save(os.path.join(out_path, str(test[i-1]))+'.npy', file_np)

def preprocess6(path, feature_set, split_path, out_path):

    train = pd.read_csv(os.path.join(split_path, 'train_split.csv'), header=0)['Participant_ID'].tolist()
    dev = pd.read_csv(os.path.join(split_path, 'dev_split.csv'), header=0)['Participant_ID'].tolist()
    test = pd.read_csv(os.path.join(split_path, 'test_split.csv'), header=0)['Participant_ID'].tolist()

    for i in range(1, len(train)+1):
        fname = 'training_' + str(i).zfill(3) + "_" + feature_set
        file_np = pd.read_csv(os.path.join(path, fname), header=None, sep=';').iloc[:,2:].values
        file_np = smooth_subsample_frames(file_np)
        np.save(os.path.join(out_path, str(train[i-1]))+'.npy', file_np)
    for i in range(1, len(dev)+1):
        fname = 'development_' + str(i).zfill(2) + "_" + feature_set
        file_np = pd.read_csv(os.path.join(path, fname), header=None, sep=';').iloc[:,2:].values
        file_np = smooth_subsample_frames(file_np)
        np.save(os.path.join(out_path, str(dev[i-1]))+'.npy', file_np)
    for i in range(1, len(dev)+1):
        fname = 'test_' + str(i).zfill(2) + "_" + feature_set
        file_np = pd.read_csv(os.path.join(path, fname), header=None, sep=';').iloc[:,2:].values
        file_np = smooth_subsample_frames(file_np)
        np.save(os.path.join(out_path, str(test[i-1]))+'.npy', file_np)

if __name__ == "__main__":

    #directory where all data is located
    source_path = "./data"
    #directory to save the .npy files
    out_path = './Avec_features'
    #directory where the train/validation/test splits are located
    split_path = './splits/'


    feature_sets = ['OpenSMILE2.3.0_mfcc.csv', 'OpenSMILE2.3.0_egemaps.csv', 'OpenFace2.1.0_Pose_gaze_AUs.csv',
                    "BoAW_openSMILE_2.3.0_eGeMAPS.csv", "BoAW_openSMILE_2.3.0_MFCC.csv", "BoVW_openFace_2.1.0_Pose_Gaze_AUs.csv",
                    'CNN_ResNet.mat', 'CNN_VGG.mat', "densenet201.csv", "vgg16.csv", ]
    dirs = ['mfcc', 'eGeMAPS', 'AUpose', 'BoW_eGeMAPS', 'BoW_mfcc', 'BoW_AUpose', 'ResNet', 'VGG', 'DS_densenet', 'DS_VGG']
    modalities = ['speech', 'speech', 'vision', 'speech', 'speech', 'vision', 'vision', 'vision', 'speech', 'speech']
    preprocess = [preprocess1, preprocess1, preprocess1, preprocess2, preprocess2, preprocess2, preprocess3, preprocess4, preprocess5, preprocess6]

    #Adjust this list based on the directory structure - directory where each feature set is located
    paths = ['Avec_data', 'Avec_data', 'Avec_data', 'audio_BoAW', 'audio_BoAW', 'BoVW', 'Avec_data', 'Avec_data', 'audio_deep_features', 'audio_deep_features']

    for fs, modality, func, dir, feature_path in zip(feature_sets, modalities, preprocess, dirs, paths):
        if not os.path.exists(os.path.join(out_path, modality, dir)):
            os.makedirs(os.path.join(out_path, modality, dir)
        func(os.path.join(source_path,feature_path), fs, split_path, os.path.join(out_path, modality, dir))
