import sys, os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as Funcs
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ModelWrapper():
    def __init__(self, models:[nn.Module], loadFromPath=None, numModels=2, tabuList=[], learningRate=0.001, criterion="CCC-cov", shuffle=True, device='cpu'):
        self.models = models
        if not torch.cuda.is_available(): device='cpu'
        self.deviceRaw = device
        self.device = torch.device(device)
        print("running on", device)
        if not loadFromPath is None: self.load_model(loadFromPath, numModels=numModels)
        self.shuffle = shuffle
        self.tabuList = tabuList
        self.evals = []
        torch.manual_seed(1)
        if not device == "cpu": torch.cuda.manual_seed(1)
        params = []
        for i, model in enumerate(self.models):
            self.models[i].to(device=self.device)
            if i in tabuList: continue
            params += model.parameters()
        self.optimizer = optim.Adam(params, lr=learningRate)
        if criterion=="CCC-cov":
            from DeepModels import CCC
            self.criterion = CCC(mode="cov")
        elif criterion=="CCC-cor":
            from DeepModels import CCC
            self.criterion = CCC(mode="cor")
        else:
            self.criterion = torch.nn.MSELoss()

    def load_model(self, filePath, numModels=-1):
        if numModels == -1: numModels = len(self.models)
        for i in range(numModels):
            if len(self.models) == numModels:
                self.models[i] = torch.load(filePath + "_" + str(i) + ".pth", map_location=self.deviceRaw)
            else:
                self.models.append(torch.load(filePath + "_" + str(i) + ".pth", map_location=self.deviceRaw))
            
    def save_model(self, savePath, fileName):
        if not os.path.exists(savePath): os.makedirs(savePath)
        for i, model in enumerate(self.models):
            torch.save(model, os.path.join(savePath, fileName + "_" + str(i) + ".pth"))

    def train(self, dataset:Dataset, batchSize=1, epochs=10, firstEpoch=1, savePath="models/first", evalDataset:Dataset=None, csvPath="models/train.csv", 
              earlyStopAfter=10, tolerance=15, computeLossFor=10 ,computeLossForEval=10, tarsFunc=lambda tars: tars[:,0], plusTar=-1):
        if firstEpoch == 1: self.evals = []
        earlyStopped = False
        for epoch in range(firstEpoch, epochs+1):
            accuracy, epochLoss = self.trainCompute(dataset=dataset, computeLossFor=computeLossFor, tarsFunc=tarsFunc, plusTar=plusTar)
            self.save_model(savePath, str(epoch)+"")
            # self.save_model(savePath, "best")
            if not evalDataset is None: 
                EvalAccuracy, evalLoss = self.testCompute(evalDataset, verbose=True, computeLossFor=computeLossForEval, tarsFunc=tarsFunc, plusTar=plusTar)
                self.writeTrainEvalOnCSV(csvPath, int(epoch), accuracy, epochLoss, EvalAccuracy, evalLoss)
                sys.stdout.write("\repoch %d completed. - accuracy: %d%% - loss: %f - dev accuracy: %d%% - dev loss: %f\n" % (epoch, accuracy, epochLoss, EvalAccuracy, evalLoss))
                if earlyStopAfter > 0:
                    self.evals.append(evalLoss)
                    bestEpoch = self.evals.index(min(self.evals))+1
                    if min(self.evals) == evalLoss: self.save_model(savePath, "best")
                    if (len(self.evals) - bestEpoch >= tolerance) and epoch > earlyStopAfter: 
                        print("Stopped early. Best loss was at epoch", bestEpoch)
                        # self.load_model(os.path.join(savePath, str(bestEpoch)))
                        # self.save_model(savePath, "best")
                        earlyStopped = True
                        break
            else:
                self.save_model(savePath, str(epoch)+"")
                sys.stdout.write("\repoch %d completed. - accuracy: %d%% - loss: %f\n" % (epoch, accuracy, epochLoss))
        return earlyStopped
    
    def testConfMatrix(self, dataset:Dataset, batchSize=1, verbose=True, numTargets=10, classFunc=lambda y: int(y)-1, tarsFunc=lambda tars: tars[:,0], plusTar=-1):
        test_loader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=self.shuffle)
        for i, model in enumerate(self.models):
            self.models[i].eval()
        confusion = np.zeros((numTargets,numTargets))
        for  batch_idx, (data, target) in enumerate(test_loader):
            if verbose: sys.stdout.write("\rTesting on given dataset %d%%" % (int(100*batch_idx*len(data)/len(dataset))))
            tars = target
            target = tarsFunc(target)
            output, (data, target) = self.getOutput(data, target)
            tar0 = tars[:,plusTar].to(device=self.device)
            output, target = output + tar0, target + tar0
            preds = output.data.round().squeeze(1)
            # print("preds:",preds)
            # print("target:",target)
            for i,targ in enumerate(target):
                # print(i, preds[i].cpu().numpy())
                # print(targ.cpu().numpy(), preds[i].cpu().numpy(), i, targ)
                pred = classFunc(preds[i].cpu().numpy())
                if pred < 0: pred=0
                if pred >= numTargets: pred = numTargets-1
                confusion[classFunc(targ.cpu().numpy()), pred] += 1
            # print(confusion)
            # print((preds == target).sum().cpu())
        if verbose: sys.stdout.write("\rTesting on given dataset completed\n")
        return confusion

    def trainCompute(self, dataset:Dataset, batchSize=1, computeLossFor=20, tarsFunc=lambda tars: tars[:,0], plusTar=-1):
        train_loader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=self.shuffle)
        for i, model in enumerate(self.models):
            self.models[i].train()
        return self.computeLossFor(train_loader, computeLossFor, tarsFunc=tarsFunc, plusTar=plusTar)

    def testCompute(self, dataset:Dataset, batchSize=1, verbose=False, computeLossFor=20, tarsFunc=lambda tars: tars[:,0], plusTar=-1):
        test_loader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=self.shuffle)
        for i, model in enumerate(self.models):
            self.models[i].eval()
        return self.computeLossFor(test_loader, computeLossFor, tarsFunc=tarsFunc, plusTar=plusTar)

    def computeLossFor(self, data_loader, computeLossFor, tarsFunc=lambda tars: tars[:,0], plusTar=-1):
        correct = 0
        allLoss = 0
        counter = 0
        outputs, targets = torch.FloatTensor([]), torch.FloatTensor([])
        outputs, targets = outputs.to(device=self.device), targets.to(device=self.device)
        
        for batch_idx, (data, target) in enumerate(data_loader):
            sys.stdout.write("\rComputing Loss %d%%" % (int(100*batch_idx*len(data)/len(data_loader.dataset))))
            tars = target
            target = tarsFunc(target)
            output, (data, target) = self.getOutput(data, target)
                
            preds = output.data.round().squeeze(1)
            correct += preds.eq((target).data.view_as(preds)).long().cpu().sum()
            if not self.models[0].training: 
                output, target = output.data, target.data # So that memory would dump the backward connections if not training
                if plusTar >= 0: 
                    tar0 = tars[:,plusTar].to(device=self.device)
                    output, target = output + tar0, target + tar0
            outputs = torch.cat((outputs, output), 0)
            targets = torch.cat((targets, target), 0)

            if ((batch_idx+1)*len(data) % computeLossFor == 0): #or ((batch_idx+1)*len(data)==len(data_loader.dataset)) :
                # print(batch_idx*len(data), len(data), len(train_loader.dataset))
                if self.models[0].training:
                    self.optimizer.zero_grad()
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                else:
                    loss = self.criterion(outputs, targets)
                allLoss += loss*len(outputs)
                counter += len(outputs)
                outputs, targets = torch.FloatTensor([]), torch.FloatTensor([])
                outputs, targets = outputs.to(device=self.device), targets.to(device=self.device)

        # print("outputs, targets", outputs.squeeze(1), targets)
        accuracy = 100. * correct / len(data_loader.dataset)
        epochLoss = allLoss / counter #/ len(data_loader.dataset)
        return accuracy.item(), epochLoss.item()

    def getOutput(self, data, target):
        data = torch.FloatTensor(data)
        target = torch.FloatTensor(target)
        data, target = data.to(device=self.device), target.to(device=self.device)
        data, target = Variable(data), Variable(target)
        output = data
        for i, model in enumerate(self.models):
            if i in self.tabuList: continue
            output = self.models[i](output)
        return output, (data, target)

    def writeTrainEvalOnCSV(self, csvPath, epoch, accuracy, epochLoss, EvalAccuracy, evalLoss):
        import pandas as pd
        header = ["epoch", "accuracy", "epochLoss", "EvalAccuracy", "evalLoss"]
        if epoch == 1 and os.path.exists(csvPath): os.remove(csvPath)
        if not os.path.exists(csvPath): 
            data = [[epoch, accuracy, epochLoss, EvalAccuracy, evalLoss]]
            df = pd.DataFrame(data, columns = header)
            df.to_csv(csvPath, index=False)
        else:
            data = {'epoch': epoch, 'accuracy': accuracy, 'epochLoss':epochLoss, 'EvalAccuracy': EvalAccuracy, 'evalLoss':evalLoss}
            df = pd.read_csv(csvPath)
            df = df.append(data, ignore_index=True)
            df.to_csv(csvPath, index=False)

    def outputSingleSample(self, data, tabuList=[], returnOut=False):
        data = torch.FloatTensor(data).unsqueeze(0)
        data = data.to(device=self.device)  
        data = Variable(data)
        output = data
        for i, model in enumerate(self.models):
            if i in tabuList: continue
            output = self.models[i](output)
        if returnOut: return output.cpu().data.numpy()[0]
        preds = output.data.round().squeeze(1)
        return preds
