
from torch.utils.data import Dataset, DataLoader, random_split
import re
import torch

import pickle

import os
import numpy as np

class BioDataset(Dataset):
    def __init__(self, args):
        super(BioDataset, self).__init__()
        print('loading dataset...')
        x_all = []
        x_padding_all = []
        pos_id_all = []
        lbl_all = []

        files_list_all = os.listdir(args.data_path)
        for f in files_list_all:
            with open(args.data_path+f, "rb") as p:
                adata = pickle.load(p)
            x = adata['x']
            x_padding = adata['x_padding']
            pos = adata['pos_id']
            lbl = np.load(args.label_path+f[:-12]+'_labels.npy')
            lbl_onehot = np.eye(2)[lbl]
            x_all.extend(x)
            x_padding_all.extend(x_padding)
            pos_id_all.extend(pos)
            lbl_all.extend(torch.tensor(lbl_onehot))



        self.x = x_all
        self.x_padding = x_padding_all
        self.pos_id = pos_id_all
        self.lbl = lbl_all
        assert len(x_all) == len(lbl_all)
        assert len(x_padding_all ) == len(lbl_all)
        assert len(pos_id_all) == len(lbl_all)

        self.length = len(self.x)
        print('number of samples:',self.length)
    def __getitem__(self, item):
        return self.x[item][:100], self.x_padding[item][:100], self.pos_id[item][:100], self.lbl[item]

    def __len__(self):
        return self.length



def dataloader(args):
    ebddataset = BioDataset(args)

    train_size = int(args.train_rate * len(ebddataset))
    test_size = len(ebddataset) - train_size

    train_dataset, test_dataset = random_split(ebddataset, [train_size, test_size])

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=True)

    return train_data_loader,test_data_loader






