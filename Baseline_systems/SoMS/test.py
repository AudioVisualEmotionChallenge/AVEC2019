from DeepModels import GRU, fullyConnected
from DatasetLoader import DatasetLoader, Datasets
from ModelWrappers import ModelWrapper
from funcs import writeLineToCSV
import torch
import os

def  main():
    savePath = os.path.join("models","SoM_GRU_1")
    datasetLoader = DatasetLoader()
    datasetLoader.dataset = Datasets().audio_features_mfcc_functionals
    datasetLoader.loadDataset()
    datasetLoader.colomnsSeeked = ["Valance", "Valence_0"]
    tarsFunc = lambda tars: tars[:,0] - tars[:,1] # the target for which the model will get trained. Depends on how it is loaded from the dataset!

    saveDescription = datasetLoader.dataset
        
    wrapper = ModelWrapper([], numModels=2, loadFromPath=os.path.join(savePath, "best"), device='cuda:0')

    _, evalLoss = wrapper.testCompute(datasetLoader.devDataset, verbose=True, computeLossFor=len(datasetLoader.devDataset), tarsFunc=tarsFunc, plusTar=-1)
    _, evalLoss2 = wrapper.testCompute(datasetLoader.devDataset, verbose=True, computeLossFor=len(datasetLoader.devDataset), tarsFunc=tarsFunc, plusTar=1)
    _, testLoss = wrapper.testCompute(datasetLoader.testDataset, verbose=True, computeLossFor=len(datasetLoader.testDataset), tarsFunc=tarsFunc, plusTar=-1)
    _, testLoss2 = wrapper.testCompute(datasetLoader.testDataset, verbose=True, computeLossFor=len(datasetLoader.testDataset), tarsFunc=tarsFunc, plusTar=1)

    writeLineToCSV(os.path.join("models","results.csv"), 
    ["savePath", "saveDescription", "evalLoss", "evalLoss2", "evalCCC", "evalCCC2", "testLoss", "testLoss2", "testCCC", "testCCC2"],
    [savePath,    saveDescription,   evalLoss,   evalLoss2, 1-evalLoss, 1-evalLoss2,  testLoss,   testLoss2, 1-testLoss, 1-testLoss2])

if __name__ == "__main__":
    main()
