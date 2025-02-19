from sklearn.model_selection import KFold

from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import re
import torch
from tqdm import tqdm
import pandas as pd
import pickle
import os
import numpy as np

class BioDataset(Dataset):
    def __init__(self, args, data_list):
        super(BioDataset, self).__init__()
        print('loading dataset...')
        ebd_all = []
        lbl_all = []
        if data_list[0]=='all':
            files_list = os.listdir(args.data_path)
            for f in files_list:
                with open(args.data_path + f, 'rb') as PKLfile:
                    PKLdata = pickle.load(PKLfile)
                ebd = PKLdata['input_ids']
                lbl = PKLdata['labels']
                lbl_onehot = np.eye(2)[lbl]
                ebd_all.extend(torch.tensor(ebd))
                lbl_all.extend(torch.tensor(lbl_onehot))
        else:
            files_list_all = os.listdir(args.data_path)
            for dt in data_list:
                for f in files_list_all:
                    if dt in f:
                        with open(args.data_path+f, 'rb') as PKLfile:
                            PKLdata = pickle.load(PKLfile)
                        ebd = PKLdata['input_ids']
                        lbl = PKLdata['labels']
                        lbl_onehot = np.eye(2)[lbl]
                        ebd_all.extend(torch.tensor(ebd))
                        lbl_all.extend(torch.tensor(lbl_onehot))


        self.embeds = ebd_all
        self.labels = lbl_all
        assert len(ebd_all) == len(lbl_all)

        self.length = len(self.embeds)
        print('number of samples:',self.length)
    def __getitem__(self, item):
        return self.embeds[item], self.labels[item]

    def __len__(self):
        return self.length



def KfoldDataset(args,data_list):
    biodataset = BioDataset(args,data_list)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    trdt_list = []
    tedt_list = []

    for train_indices, val_indices in kf.split(biodataset):
        train_dataset = torch.utils.data.Subset(biodataset, train_indices)
        test_dataset = torch.utils.data.Subset(biodataset, val_indices)
        trdt_list.append(train_dataset)
        tedt_list.append(test_dataset)


    return trdt_list, tedt_list




def dataloader(current_fold,train_list,test_list,tr_bs,te_bs):
    train_data_loader = DataLoader(dataset=train_list[current_fold], batch_size=tr_bs, shuffle=True)
    test_data_loader = DataLoader(dataset=test_list[current_fold], batch_size=te_bs, shuffle=True)

    return train_data_loader,test_data_loader






