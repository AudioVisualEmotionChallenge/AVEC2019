import sys, os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from random import randint
import pandas as pd
from read_csv import load_features

class speechDataset(Dataset):
    '''
    # read wav files and use for ML purposes (built on top of Dataset class of pytorch)
    # path is where the folders of classes exists and each folder of class contains .wav files.
    # features are "csvFeats" or "None"
    # obviously this will not work for loading batches of data if files do not have the same length!
    --> exp: dataset = speechDataset("./data/segments/", "./data/train.csv", classFunc=lambda y: int((y-2)/3))
    '''
    def __init__(self, audioPath, csvPath, filesExt=".csv", delim=';', classFunc=lambda y: y, sr=None, WinSize=0.025, WinShift=0.010, n_mels=128, shouldNormalize=True,
                 colomnNames="File", colomnsSeeked=["Valance", "Valence_0"], labeldtype="f", noTimeStamps=False, skip_header=True, skip_instname=True):
        super(speechDataset, self).__init__()
        self.audioPath = audioPath
        self.csvPath = csvPath
        self.filesExt = filesExt
        self.delim = delim
        self.sr = sr
        self.n_mels = n_mels
        self.classFunc = classFunc
        self.WinSize = WinSize
        self.WinShift = WinShift
        self.shouldNormalize = shouldNormalize
        self.colomnNames = colomnNames
        self.colomnsSeeked = colomnsSeeked
        self.labeldtype = labeldtype
        self.noTimeStamps = noTimeStamps
        self.skip_header = skip_header
        self.skip_instname = skip_instname
        self.create_paths_and_labels()

    def __len__(self):
        return len(self.filePaths)

    def __getitem__(self, idx):
        path = self.filePaths[idx]
        labels = self.labels[idx]
        frames = load_features(path, skip_header=self.skip_header, skip_instname=self.skip_instname, delim=self.delim)
        if self.noTimeStamps:
            frames = frames[:,1:]
        if self.shouldNormalize:
            frames = self.normalize(frames)
        frames = np.array(frames, dtype='f')
        # print(frames)
        return frames, np.array(labels, dtype=self.labeldtype)

    def shape(self):
        frames, _ = self[0]
        return (frames.shape)

    def create_paths_and_labels(self):
        df = pd.read_csv(self.csvPath)
        my_list = df[self.colomnNames].values
        self.filePaths = []
        self.labels = []
        filesCount = len(my_list)
        for i, fileName in enumerate(my_list):
            sys.stdout.write("\rpreparing files %d%%" % int(100*i/(filesCount)))
            fileNameExt = fileName + self.filesExt
            self.filePaths.append(os.path.join(self.audioPath, fileNameExt))
            labels = []
            for colomnSeeked in self.colomnsSeeked:
                label = df[i:i+1][colomnSeeked]
                label = self.classFunc(label)
                labels.append(label)
            self.labels.append(labels)
        sys.stdout.write("\rpreparing files completed\n")

    def normalize(self, feats):
        result = feats
        if (feats.std(axis=0) != 0).all(): result = (feats - feats.mean(axis=0)) / feats.std(axis=0)
        return result
