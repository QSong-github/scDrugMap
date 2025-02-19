"""
Dataloaders

"""

import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append('../')

import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, random_split
import os
from torch.utils.data import Dataset

CLASS = 5 + 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class SCDataset(Dataset):
    def __init__(self, dt_path,label_path):
        super().__init__()
        self.samples = np.load(dt_path)
        self.labels = np.load(label_path)
        pass

    def __getitem__(self, index):
        #rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.samples[index].astype(float)
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()[:6000]
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        labels = self.labels[index]
        lbl_onehot = np.eye(2)[labels]
        lbl_onehot = torch.tensor(lbl_onehot)
        return full_seq, lbl_onehot

    def __len__(self):
        return self.samples.shape[0]

def buildDataset(args):
    dt_all = []
    files_list_all = os.listdir(args.data_path)
    for f in files_list_all:
        dt_path = args.data_path + f
        label_path = args.label_path + f[:-11] + 'labels.npy'
        dataset = SCDataset(dt_path,label_path)
        dt_all.append(dataset)


    # 使用ConcatDataset来合并它们
    merged_dataset = ConcatDataset(dt_all)
    train_size = int(args.train_rate * len(merged_dataset))
    test_size = len(merged_dataset) - train_size

    train_dataset, test_dataset = random_split(merged_dataset, [train_size, test_size])



    return train_dataset, test_dataset




def dataloader(args):
    train_dataset, test_dataset = buildDataset(args)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False)

    return train_data_loader,test_data_loader
