import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class myDataset(Dataset):
    def __init__(self, address="data.csv", tars=[1,2], classFunc=lambda y: y):
        super(myDataset, self).__init__()
        self.data = pd.read_csv(address)
        self.cols = self.data.keys()
        self.classFunc = classFunc
        self.npdata = self.data.values
        self.tars = tars
        self.len = self.data.shape[0]

    def __getitem__(self, index):
        inputs = np.array(self.npdata[index, 2:], dtype=float)
        output = []
        for tar in self.tars:
            output.append(self.classFunc(self.npdata[index, tar]))
        return torch.from_numpy(inputs).float(), torch.from_numpy(np.array(output, dtype="f"))

    def __len__(self):
        return self.len

    def shape(self):
        frames, _ = self[0]
        return (frames.shape)
        

# dataset = myDataset()
# train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
# print(train_loader)
# for inputs, labels in train_loader:
#     print(inputs, labels);break
