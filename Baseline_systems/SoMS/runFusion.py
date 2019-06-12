from DeepModels import fullyConnected
from DatasetLoader import DatasetLoader, Datasets
from ModelWrappers import ModelWrapper
from funcs import UAR_from_ConfMat, plot_confusion, writeLineToCSV
from data_loader import myDataset
from speechDataset import speechDataset
import os

def main():
    savePath = os.path.join("models","SoM_Mix_1")
    saveDescription = "mix: 1-2"
    trainDatasets, devDatasets, testDatasets = [], [], []
    
    ### dataset 1 #########
    datasetLoader1 = DatasetLoader()
    datasetLoader1.dataset = Datasets().audio_features_mfcc_functionals
    datasetLoader1.loadDataset()
    trainDatasets.append(datasetLoader1.trainDataset)
    devDatasets.append(datasetLoader1.devDataset)
    testDatasets.append(datasetLoader1.testDataset)
    #######################
    ### dataset 2 #########
    datasetLoader2 = DatasetLoader()
    datasetLoader2.dataset = Datasets().visual_features_functionals
    datasetLoader2.loadDataset()
    trainDatasets.append(datasetLoader2.trainDataset)
    devDatasets.append(datasetLoader2.devDataset)
    testDatasets.append(datasetLoader2.testDataset)
    #######################

    ### models ############
    model1Path = os.path.join("models","SoM_GRU_1","best")
    model2Path = os.path.join("models","SoM_GRU_2","best")
    models = [model1Path, model2Path]
    #######################
    
    # the paths to where the fused features would be (or already are)
    trainPath = os.path.join(savePath, "trainData.csv")
    devPath = os.path.join(savePath, "devData.csv")
    testPath = os.path.join(savePath, "testData.csv")

    # comment out the next three lines if already got the CSV files of fused feats for train
    modelsOutToCSVs(models, trainDatasets, trainPath)
    modelsOutToCSVs(models, devDatasets, devPath)
    modelsOutToCSVs(models, testDatasets, testPath)

    trainDataset = myDataset(address=trainPath, tars=[1,2])
    devDataset = myDataset(address=devPath, tars=[1,2])
    testDataset = myDataset(address=testPath, tars=[1,2])
    tarsFunc = lambda tars: tars[:,0] - tars[:,1] # the target for which the model will get trained. Depends on how it is loaded from the dataset!

    featSize = trainDataset.shape()[-1]
    model = fullyConnected(featSize, 1, hiddenSize=32)
    wrapper = ModelWrapper([model], tabuList=[], device='cuda:0')
    # comment out the next lines if you just want to test
    wrapper.train(trainDataset, epochs=2500, firstEpoch=1, savePath=savePath, evalDataset=devDataset, csvPath=os.path.join(savePath, "trainLog.csv"), 
                    computeLossFor=len(trainDataset), computeLossForEval=len(devDataset), tolerance=5, tarsFunc=tarsFunc, plusTar=-1)

    wrapper.load_model(os.path.join(savePath, "best"))
    _, evalLoss = wrapper.testCompute(devDataset, verbose=True, computeLossFor=len(devDataset), tarsFunc=tarsFunc, plusTar=-1)
    _, evalLoss2 = wrapper.testCompute(devDataset, verbose=True, computeLossFor=len(devDataset), tarsFunc=tarsFunc, plusTar=1)
    _, testLoss = wrapper.testCompute(testDataset, verbose=True, computeLossFor=len(testDataset), tarsFunc=tarsFunc, plusTar=-1)
    _, testLoss2 = wrapper.testCompute(testDataset, verbose=True, computeLossFor=len(testDataset), tarsFunc=tarsFunc, plusTar=1)

    writeLineToCSV(os.path.join("models","results.csv"), 
    ["savePath", "saveDescription", "evalLoss", "evalLoss2", "evalCCC", "evalCCC2", "testLoss", "testLoss2", "testCCC", "testCCC2"], 
    [savePath,    saveDescription,   evalLoss,   evalLoss2, 1-evalLoss, 1-evalLoss2,  testLoss,   testLoss2, 1-testLoss, 1-testLoss2])

def modelsOutToCSVs(models, datasets, csvPath):
    wrappers = []
    for model in models:
        wrapper = ModelWrapper([], loadFromPath=model, tabuList=[1], device='cuda:0')
        wrappers.append(wrapper)
    writeToCSV(wrappers, datasets, csvPath, tabuList=[1])

def writeToCSV(wrappers:[ModelWrapper], datasets:[speechDataset], csvPath, tabuList=[], classFunc = lambda y: int(y)+1, tarsIdx=[0,1]):
    import pandas as pd
    import sys
    data = []
    headers = []
    for r, wrapper in enumerate(wrappers):
        dataset = datasets[r]
        for i in range(len(dataset)):
            sys.stdout.write("\rRunning model %d on the given data %d%%" % (r, int(100*i/len(dataset))))
            inputData = dataset[i][0]
            fileName = dataset.filePaths[i]
            lastSlashPos = fileName.rfind(os.path.split(fileName)[-1])-1
            output = wrapper.outputSingleSample(inputData, returnOut=True, tabuList=tabuList)
            # print(output)
            targets = []
            for tarId in tarsIdx:
                targets.append(dataset[i][1][tarId])
            # print(valence, valence0, dataset[i][1])
            
            if r > 0:
                dic = data[i] 
            else:
                dic = {'file': fileName[lastSlashPos+1:]}
                for j, target in enumerate(targets):
                    dic['target_'+str(j)] = classFunc(target)
            if i==0 and r==0: 
                headers.append('file')
                for j, target in enumerate(targets): headers.append('target_'+str(j))
            for j, out in enumerate(output): 
                head = 'output_'+str(r)+"_"+str(j)
                if i==0: headers.append(head)
                dic[head] = out
            if r > 0:
                data[i] = dic
            else:
                data.append(dic)
    sys.stdout.write("\rRunning models on the given data completed.\n")
    df = pd.DataFrame(data, columns = headers)
    pathDir = csvPath[:csvPath.rfind(os.path.split(csvPath)[-1])]
    if not os.path.exists(pathDir): os.makedirs(pathDir)
    df.to_csv(csvPath, index=False)

if __name__ == "__main__":
    main()
