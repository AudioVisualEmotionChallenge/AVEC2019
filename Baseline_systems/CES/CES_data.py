# Load features for all partitions and labels for training and development partitions for AVEC 2019 CES (see function load_CES_data(...))
import pandas as pd
import numpy as np

def load_features(path_features, partition, num_inst, max_seq_len):
    # Check for header and separator
    with open(path_features + partition + '_01.csv') as infile:
        line = infile.readline()
    header = None
    if line[:4]=='name':
        header = 'infer'
    sep=';'
    if ',' in line:
        sep=','
    
    # Read feature files
    num_features = len(pd.read_csv(path_features + partition + '_01.csv', sep=sep, header=header).columns) - 2  # do not consider instance name and time stamp
    features = np.empty((num_inst, max_seq_len, num_features))
    for n in range(0, num_inst):
        F = pd.read_csv(path_features + partition + '_' + str(n+1).zfill(2) + '.csv', sep=sep, header=header, usecols=range(2,2+num_features)).values
        if F.shape[0]>max_seq_len: F = F[:max_seq_len,:]  # might occur for some feature representations
        features[n,:,:] = np.concatenate((F, np.zeros((max_seq_len - F.shape[0], num_features))))  # zero padded
    return features

def load_labels(path_labels, partition, num_inst, max_seq_len, targets):
    labels_original = []
    labels_padded   = []
    for tar in targets:
        labels_original_tar = []
        labels_padded_tar   = np.empty((num_inst, max_seq_len, 1))
        for n in range(0, num_inst):
            yn = pd.read_csv(path_labels + partition + '_' + str(n+1).zfill(2) + '.csv', sep=';', usecols=[tar]).values
            labels_original_tar.append(yn)  # original length sequence
            labels_padded_tar[n,:,:] = np.concatenate((yn, np.zeros((max_seq_len - yn.shape[0], 1))))  # zero padded
        labels_original.append(labels_original_tar)
        labels_padded.append(labels_padded_tar)
    return labels_original, labels_padded

def load_CES_data(base_folder='../', features_folders=['audio_features_egemaps_xbow/', 'visual_features_xbow/'], targets=['arousal','valence','liking'], test_available=False):
    # Load the AVEC 2019 CES (SEWA) data. German and Hungarian training and development sets, and German, Hungarian, and Chinese test sets.
    # 
    # base_folder:      path to the folder with features and labels files for AVEC 2019 CES (including a '/' at the end); features must have the same frequency/hop size as the labels (=0.1s for CES)
    # features_folders: list of folders below the base folder, where the features should be loaded (including a '/' at the end of each folder name)
    # targets:          list of targets to return as labels, default: ['arousal','valence','liking']
    # test_available:   boolean, set True only, if the features for the test partitions are available in the corresponding feature folders. If false, the corresponding objects will be empty.
    # 
    # Returns: 13-tuple:
    #  1) train_DE_x: numpy-array (number of subjects, maximum sequence length, number of features) of the features for German-Train; short sequences are padded with zeros
    #  2) train_HU_x: numpy-array (number of subjects, maximum sequence length, number of features) of the features for Hungarian-Train; short sequences are padded with zeros
    #  3) train_DE_y: list with a numpy-array (number of subjects, maximum sequence length, 1) for each target (arousal, valence, liking) for German-Train; short sequences are padded with zeros
    #  4) train_HU_y: list with a numpy-array (number of subjects, maximum sequence length, 1) for each target (arousal, valence, liking) for Hungarian-Train; short sequences are padded with zeros
    #  5) devel_DE_x: numpy-array (number of subjects, maximum sequence length, number of features) of the features for German-Development; short sequences are padded with zeros
    #  6) devel_HU_x: numpy-array (number of subjects, maximum sequence length, number of features) of the features for Hungarian-Development; short sequences are padded with zeros
    #  7) devel_DE_labels_original: list with a list for each target (arousal, valence, liking), where the inner list contains a numpy-array (sequence length, 1) with the labels for each subject; short sequences are not padded, so that the evaluation can be done on the original sequence; for German-Development
    #  8) devel_HU_labels_original: list with a list for each target (arousal, valence, liking), where the inner list contains a numpy-array (sequence length, 1) with the labels for each subject; short sequences are not padded, so that the evaluation can be done on the original sequence; for Hungarian-Development
    #  9) test_DE_x: numpy-array (number of subjects, maximum sequence length, number of features) of the features for German-Test; short sequences are padded with zeros
    # 10) test_HU_x: numpy-array (number of subjects, maximum sequence length, number of features) of the features for Hungarian-Test; short sequences are padded with zeros
    # 11) test_CN_x: numpy-array (number of subjects, maximum sequence length, number of features) of the features for Chinese-Test; short sequences are padded with zeros
    
    # Number of recordings
    num_train_DE = 34
    num_train_HU = 34
    num_devel_DE = 14
    num_devel_HU = 14
    
    num_test_DE = 16
    num_test_HU = 18
    num_test_CN = 70
    
    max_seq_len = 1768  # Maximum number of labels in one sequence
    
    # Initialise numpy arrays
    train_DE_x = np.empty((num_train_DE, max_seq_len, 0))
    train_HU_x = np.empty((num_train_HU, max_seq_len, 0))
    devel_DE_x = np.empty((num_devel_DE, max_seq_len, 0))
    devel_HU_x = np.empty((num_devel_HU, max_seq_len, 0))
    
    test_DE_x = np.empty((num_test_DE, max_seq_len, 0))
    test_HU_x = np.empty((num_test_HU, max_seq_len, 0))
    test_CN_x = np.empty((num_test_CN, max_seq_len, 0))
    
    for features_folder in features_folders:
        train_DE_x = np.concatenate( (train_DE_x, load_features(path_features=base_folder+features_folder, partition='Train_DE', num_inst=num_train_DE, max_seq_len=max_seq_len) ), axis=2)
        train_HU_x = np.concatenate( (train_HU_x, load_features(path_features=base_folder+features_folder, partition='Train_HU', num_inst=num_train_HU, max_seq_len=max_seq_len) ), axis=2)
        devel_DE_x = np.concatenate( (devel_DE_x, load_features(path_features=base_folder+features_folder, partition='Devel_DE', num_inst=num_devel_DE, max_seq_len=max_seq_len) ), axis=2)
        devel_HU_x = np.concatenate( (devel_HU_x, load_features(path_features=base_folder+features_folder, partition='Devel_HU', num_inst=num_devel_HU, max_seq_len=max_seq_len) ), axis=2)
        if test_available:
            test_DE_x = np.concatenate( (test_DE_x, load_features(path_features=base_folder+features_folder, partition='Test_DE', num_inst=num_test_DE, max_seq_len=max_seq_len) ), axis=2)
            test_HU_x = np.concatenate( (test_HU_x, load_features(path_features=base_folder+features_folder, partition='Test_HU', num_inst=num_test_HU, max_seq_len=max_seq_len) ), axis=2)
            test_CN_x = np.concatenate( (test_CN_x, load_features(path_features=base_folder+features_folder, partition='Test_CN', num_inst=num_test_CN, max_seq_len=max_seq_len) ), axis=2)
    
    _                       , train_DE_y = load_labels(path_labels=base_folder+'labels/', partition='Train_DE', num_inst=num_train_DE, max_seq_len=max_seq_len, targets=targets)
    _                       , train_HU_y = load_labels(path_labels=base_folder+'labels/', partition='Train_HU', num_inst=num_train_HU, max_seq_len=max_seq_len, targets=targets)
    devel_DE_labels_original, _          = load_labels(path_labels=base_folder+'labels/', partition='Devel_DE', num_inst=num_devel_DE, max_seq_len=max_seq_len, targets=targets)
    devel_HU_labels_original, _          = load_labels(path_labels=base_folder+'labels/', partition='Devel_HU', num_inst=num_devel_HU, max_seq_len=max_seq_len, targets=targets)
    
    return train_DE_x, train_HU_x, train_DE_y, train_HU_y, devel_DE_x, devel_HU_x, devel_DE_labels_original, devel_HU_labels_original, test_DE_x, test_HU_x, test_CN_x

