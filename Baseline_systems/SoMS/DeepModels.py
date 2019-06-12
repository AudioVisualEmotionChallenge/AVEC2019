import argparse
import torch
import torch.nn as nn
import torch.nn.functional as Funcs
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def get_flat_fts(self, featSize, fts):
        in_size = (1, 1, featSize)
        f = fts(Variable(torch.ones(in_size)))
        # print("f.size()",f.size())
        # print("in_size",in_size)
        return int(np.prod(f.size()[1:]))
            


class fullCNN(BaseModel):
    def __init__(self, featSize, numTargets = 3):
        super(fullCNN, self).__init__()
        self.numTargets = numTargets
        self.features = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=100, stride=50),
            nn.MaxPool1d((2)),
            nn.ReLU(),
            # nn.Conv1d(10, 20, kernel_size=5),
            # nn.MaxPool1d((2)),
            # nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.flat_fts = self.get_flat_fts(featSize, self.features)
        # self.rnn = nn.LSTM(
        #     input_size=self.flat_fts, 
        #     hidden_size=256, 
        #     num_layers=1,
        #     batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.flat_fts, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, numTargets),
            nn.LogSoftmax()
        )
        # if self.args.cuda: self.cuda()
    
    def forward(self, x):
        batch_size, in_channels, timesteps, sq_len = x.size()
        x = x.view(batch_size * timesteps, in_channels, sq_len)
        x = self.features(x)
        # print(x.size(0), self.flat_fts)
        # r_in = x.view(batch_size, timesteps, self.flat_fts)
        # r_out, (h_n, h_c) = self.rnn(r_in)
        # print(r_out.size())
        # x = self.classifier(r_out[:, -1, :])
        x = self.classifier(x.view(batch_size, self.flat_fts))
        return x

class CNNGRU(BaseModel):
    def __init__(self, featSize, numTargets = 3, num_filters=20, gru_hidden_size=128, gru_num_layers=2, mlp_hidden_size=32):
        super(CNNGRU, self).__init__()
        self.numTargets = numTargets
        self.features = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=30, stride=10),
            nn.MaxPool1d((2)),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=10, stride=5),
            nn.MaxPool1d((2)),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.flat_fts = self.get_flat_fts(featSize, self.features)
        print("conv output size:", self.flat_fts)
        self.rnn = nn.GRU(
            input_size=self.flat_fts, 
            hidden_size=gru_hidden_size, 
            num_layers=gru_num_layers,
            batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(gru_hidden_size, mlp_hidden_size),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),
            # nn.Linear(128, 64),
            # nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, numTargets),
            # nn.LogSoftmax(dim=1)
        )
        # if self.args.cuda: self.cuda()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        batch_size, in_channels, timesteps, sq_len = x.size()
        x = x.view(batch_size * timesteps, in_channels, sq_len)
        x = self.features(x)
        r_in = x.view(batch_size, timesteps, self.flat_fts)
        r_out, h_n = self.rnn(r_in)
        x = self.classifier(r_out[:, -1, :])
        return x


class featTransform(nn.Module):
    def __init__(self, in_size, out_size, dropOut=0.1):
        super(featTransform, self).__init__()
        self.out_size = out_size
        self.transformer = nn.GRU(
            input_size=in_size, 
            hidden_size=out_size, 
            num_layers=1,
            batch_first=True)
    def forward(self, x):
        batch_size, timesteps, sq_len = x.size()
        # inp = x.view(batch_size * timesteps, sq_len)
        out, _ = self.transformer(x)
        return out.view(batch_size, timesteps, self.out_size)

import random
class GRU(nn.Module):
    def __init__(self, featSize, gru_hidden_size=128, gru_num_layers=2, dropOut=0.1):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(
            input_size=featSize, 
            hidden_size=gru_hidden_size, 
            num_layers=gru_num_layers,
            batch_first=True)
        self.dropout = nn.Dropout(p=dropOut)
    
    def forward(self, x):
        batch_size, timesteps, sq_len = x.size()
        r_in = x.view(batch_size, timesteps, sq_len)
        r_out, h_n = self.rnn(r_in)
        myOut = r_out[:, -1, :]
        myOut = self.dropout(myOut)
        return myOut


class fullyConnected(nn.Module):
    def __init__(self, inSize, outSize, hiddenSize=32, dropOut=0):
        super(fullyConnected, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(inSize, hiddenSize),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropOut),
            nn.Linear(hiddenSize, outSize),
        )
    
    def forward(self, x):
        myOut = self.classifier(x)
        return myOut

class CCC(nn.Module):
    # mode == "cor" or "cov" -> "cor" for using correlation based approach and "cov" for using the covariance based approach of implementation
    def __init__(self, mode="cov"):
        super(CCC, self).__init__()
        self.mode = mode

    def forward(self, prediction, ground_truth):
        # ground_truth = (ground_truth == torch.arange(self.num_classes).cuda().reshape(1, self.num_classes)).float()
        # ground_truth = ground_truth.squeeze(0)
        prediction = prediction.squeeze(1)
        # print("")
        # print("ground_truth", ground_truth)
        # print("prediction", prediction)
        mean_gt = torch.mean (ground_truth, 0)
        mean_pred = torch.mean (prediction, 0)
        var_gt = torch.var (ground_truth, 0)
        var_pred = torch.var (prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
        if self.mode == "cor":
            cor = torch.sum (v_pred * v_gt) / (torch.sqrt(torch.sum(v_pred ** 2)) * torch.sqrt(torch.sum(v_gt ** 2)))
            sd_gt = torch.std(ground_truth)
            sd_pred = torch.std(prediction)
            numerator=2*cor*sd_gt*sd_pred
        elif self.mode == "cov":
            cov = torch.mean(v_pred * v_gt)
            numerator=2*cov
        else:
            print("mode for CCC is not defined!")
        ccc = numerator/denominator
        # print("ccc", ccc, mean_gt, mean_pred,var_gt,var_pred)
        return 1-ccc