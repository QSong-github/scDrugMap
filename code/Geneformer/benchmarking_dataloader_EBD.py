
from torch.utils.data import Dataset, DataLoader, random_split
import torch


import os
import numpy as np

class BioDataset(Dataset):
    def __init__(self, args):
        super(BioDataset, self).__init__()
        print('loading dataset...')
        ebd_all = []
        lbl_all = []
        files_list_all = os.listdir(args.data_path)
        for f in files_list_all:
            ebd = np.load(args.data_path+f)
            lbl = np.load(args.label_path+f[:-18]+'labels.npy')
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



def dataloader(args):
    ebddataset = BioDataset(args)
    # 计算训练集和验证集的大小
    train_size = int(args.train_rate * len(ebddataset))
    test_size = len(ebddataset) - train_size

    # 使用random_split进行随机分割
    train_dataset, test_dataset = random_split(ebddataset, [train_size, test_size])

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=True)

    return train_data_loader,test_data_loader






