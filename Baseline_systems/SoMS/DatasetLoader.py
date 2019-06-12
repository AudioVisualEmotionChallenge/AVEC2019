from torch.utils.data import Dataset, DataLoader
from speechDataset import speechDataset
import os

class Datasets():
    def __init__(self):
        self.audio_features_DeepSpectrum_DenseNet121 = "audio_features_DeepSpectrum_DenseNet121"
        self.audio_features_DeepSpectrum_VGG16 = "audio_features_DeepSpectrum_VGG16"
        self.audio_features_egemaps = "audio_features_egemaps"
        self.audio_features_egemaps_functionals = "audio_features_egemaps_functionals"
        self.audio_features_egemaps_xbow = "audio_features_egemaps_xbow"
        self.audio_features_mfcc = "audio_features_mfcc"
        self.audio_features_mfcc_functionals = "audio_features_mfcc_functionals"
        self.audio_features_mfcc_xbow = "audio_features_mfcc_xbow"
        self.visual_features = "visual_features"
        self.visual_features_functionals = "visual_features_functionals"
        self.visual_features_ResNet = "visual_features_ResNet"
        self.visual_features_VGG = "visual_features_VGG"
        self.visual_features_xbow = "visual_features_xbow"

class DatasetLoader():
    def __init__(self, classFunc=lambda y: int(y)-1):
        self.datasets = Datasets()
        self.dataset = self.datasets.audio_features_DeepSpectrum_DenseNet121
        self.filesPath = "/run/media/sina/207AB4977AB46AE4/datasets/AVEC2019/"
        self.trainCsvPath = os.path.join(".","data","TRAIN_SOM_AVEC2019_Dynamic_Equal.csv")
        self.devCsvPath = os.path.join(".","data","DEVELOPMENT_SOM_AVEC2019_Dynamic.csv")
        self.testCsvPath =os.path.join(".","data","TEST_SOM_AVEC2019_Dynamic.csv")
        self.colomnsSeeked = ["Valance", "Valence_0"]
        self.classFunc = classFunc
        
    def loadDataset(self):
        shouldNormalize = True
        noTimeStamps = True
        trainFolder = "Train_Audio" if self.dataset[:5] == "audio" else "Train_Visual"
        devFolder = "Development_Audio" if self.dataset[:5] == "audio" else "Development_Visual"
        testFolder = "Test_Audio" if self.dataset[:5] == "audio" else "Test_Visual"
        featsFolder = self.dataset
        trainFeatsPath = os.path.join(self.filesPath, trainFolder, featsFolder)
        devFeatsPath = os.path.join(self.filesPath, devFolder, featsFolder)
        testFeatsPath = os.path.join(self.filesPath, testFolder, featsFolder)
        # print(trainFeatsPath, self.filesPath, trainFolder, featsFolder)
        if self.dataset == self.datasets.visual_features_ResNet:
            filesExt = "_res.txt"; delim = ","; skipHeaderAndInsts = False; noTimeStamps = False
        elif self.dataset == self.datasets.visual_features_VGG:
            filesExt = "_vgg.txt"; delim = ","; skipHeaderAndInsts = False; noTimeStamps = False
        else:
            filesExt = ".csv"; delim = ";"; skipHeaderAndInsts = True

        self.trainDataset = speechDataset(trainFeatsPath, self.trainCsvPath, filesExt=filesExt, delim=delim, classFunc=self.classFunc, shouldNormalize=shouldNormalize, 
                                      noTimeStamps=noTimeStamps, skip_header=skipHeaderAndInsts, skip_instname=skipHeaderAndInsts, colomnsSeeked=self.colomnsSeeked)
        self.devDataset = speechDataset(devFeatsPath, self.devCsvPath, filesExt=filesExt, delim=delim, classFunc=self.classFunc, shouldNormalize=shouldNormalize, 
                                    noTimeStamps=noTimeStamps, skip_header=skipHeaderAndInsts, skip_instname=skipHeaderAndInsts, colomnsSeeked=self.colomnsSeeked)
        self.testDataset = speechDataset(testFeatsPath, self.testCsvPath, filesExt=filesExt, delim=delim, classFunc=self.classFunc, shouldNormalize=shouldNormalize, 
                                    noTimeStamps=noTimeStamps, skip_header=skipHeaderAndInsts, skip_instname=skipHeaderAndInsts, colomnsSeeked=self.colomnsSeeked)

