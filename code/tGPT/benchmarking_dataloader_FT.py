

from torch.utils.data import Dataset, DataLoader, random_split

import torch

import os
import numpy as np

class BioDataset(Dataset):
    def __init__(self, args):
        super(BioDataset, self).__init__()
        print('loading dataset...')
        spl_all = []
        lbl_all = []

        files_list_all = os.listdir(args.data_path)
        for f in files_list_all:
            spl = np.load(args.data_path+f, allow_pickle=True)['array']
            lbl = np.load(args.label_path+f[:-12]+'_labels.npz')['array']
            lbl_onehot = np.eye(2)[lbl]
            spl_all.extend(spl)
            lbl_all.extend(torch.tensor(lbl_onehot))


        self.samples = spl_all
        self.labels = lbl_all
        assert len(spl_all) == len(lbl_all)

        self.length = len(self.samples)
        print('number of samples:',self.length)
    def __getitem__(self, item):
        return self.samples[item], self.labels[item]

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






