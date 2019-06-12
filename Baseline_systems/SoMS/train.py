from DeepModels import GRU, fullyConnected
from DatasetLoader import DatasetLoader, Datasets
from ModelWrappers import ModelWrapper
import torch
import os

def  main():
    savePath = os.path.join("models","SoM_GRU_1")
    datasetLoader = DatasetLoader()
    datasetLoader.dataset = Datasets().audio_features_mfcc_functionals
    datasetLoader.loadDataset()
    datasetLoader.colomnsSeeked = ["Valance", "Valence_0"]
    tarsFunc = lambda tars: tars[:,0] - tars[:,1] # the target for which the model will get trained. Depends on how it is loaded from the dataset!

    curriculum      = False
    gru_hidden_size = 64
    gru_num_layers  = 2
    mlp_hidden_size = 32
    dropOut         = 0.1

    computeLossFor     = len(datasetLoader.trainDataset)
    computeLossForEval = len(datasetLoader.devDataset)
        
    featSize = datasetLoader.trainDataset.shape()[1]
    model1 = GRU(featSize, gru_hidden_size=gru_hidden_size, gru_num_layers=gru_num_layers, dropOut=dropOut)
    model2 = fullyConnected(gru_hidden_size, 1, dropOut=0)
    wrapper = ModelWrapper([model1, model2], device='cuda:0')

    # Curriculum learning or not
    if curriculum:
        from preprocess import filterClasses
        includes = [[2,3,9,10],[2,3,4,8,9,10],[1,2,3,4,5,6,7,8,9,10]] # the labeled data included for curriculum learning to get trained in order
        epochs = [32, 64, 512] # the amount of epochs to train each list included for curriculum learning, note that for the last one, early stopping would come into play
        for i, include in enumerate(includes):
            trainCSVpathDynamicEqualC = os.path.join(".","data","trainDynamicEqualC_") + str(i) + ".csv"
            filterClasses(datasetLoader.trainCsvPath, trainCSVpathDynamicEqualC, colomnTarget="Valance", includeList=include)
        for i in range(len(includes)):
            trainCSVpathDynamicEqualC = os.path.join(".","data","trainDynamicEqualC_") + str(i) + ".csv"
            datasetLoader.trainCsvPath = trainCSVpathDynamicEqualC
            datasetLoader.loadDataset()
            computeLossFor     = len(datasetLoader.trainDataset)
            computeLossForEval = len(datasetLoader.devDataset)
            first = 1 if i == 0 else epochs[i-1] + 1
            wrapper.train(datasetLoader.trainDataset, epochs=epochs[i], firstEpoch=first, savePath=savePath, evalDataset=datasetLoader.devDataset, csvPath=os.path.join(savePath, "trainLog.csv"), 
                        computeLossFor=computeLossFor, computeLossForEval=computeLossForEval, earlyStopAfter=epochs[-2]+20, tolerance=25, tarsFunc=tarsFunc)
    else:
        wrapper.train(datasetLoader.trainDataset, epochs=500, firstEpoch=1, savePath=savePath, evalDataset=datasetLoader.devDataset, csvPath=os.path.join(savePath, "trainLog.csv"), 
                    computeLossFor=computeLossFor, earlyStopAfter=60, computeLossForEval=computeLossForEval, tolerance=25, tarsFunc=tarsFunc)

if __name__ == "__main__":
    main()
