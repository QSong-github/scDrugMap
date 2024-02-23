from sklearn.model_selection import KFold
import os
from torch.utils.data import Dataset, DataLoader
import re
import torch
from tqdm import tqdm
import ast
from model import LoRAformer

from datasets import load_from_disk





class BioDataset(Dataset):
    def __init__(self, f_path):
        super(BioDataset, self).__init__()
        print('loading dataset...')
        self.dataset = load_from_disk(f_path)


        self.tokens = self.dataset['input_ids']
        self.labels_cell = self.dataset['cell_label']
        self.labels_gene = self.dataset['gene_label']

        self.length = len(self.tokens)
        print('number of sequence',len(self.tokens))
        print('number of gene',len(self.labels_gene))

    def __getitem__(self, item):
        return self.tokens[item], self.labels_cell[item], self.labels_gene[item]

    def __len__(self):
        return self.length


def bio_collate_fn(batches):
    batch_token = []
    batch_label_cell = []
    batch_label_gene = []
    for batch in batches:
        batch_token.append(torch.tensor(batch[0]))
        batch_label_cell.append(batch[1])
        batch_label_gene.append(torch.tensor(batch[2]))

    batch_token = torch.stack(batch_token)
    batch_label_gene = torch.stack(batch_label_gene)

    cell_one_hot_label = [torch.tensor([1 - item, item]) for item in batch_label_cell]
    cell_one_hot_label = torch.stack(cell_one_hot_label)


    return batch_token,cell_one_hot_label,batch_label_gene


def KfoldDataset(train_data_path, folds):
    biodataset = BioDataset(train_data_path)
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    trdt_list = []
    tedt_list = []

    for train_indices, val_indices in kf.split(biodataset):
        train_dataset = torch.utils.data.Subset(biodataset, train_indices)
        test_dataset = torch.utils.data.Subset(biodataset, val_indices)
        trdt_list.append(train_dataset)
        tedt_list.append(test_dataset)


    return trdt_list, tedt_list



def dataloader(current_fold,train_list,test_list,tr_bs,te_bs):
    train_data_loader = DataLoader(dataset=train_list[current_fold], batch_size=tr_bs, shuffle=True, collate_fn=bio_collate_fn)
    test_data_loader = DataLoader(dataset=test_list[current_fold], batch_size=te_bs, shuffle=True, collate_fn=bio_collate_fn)

    return train_data_loader,test_data_loader






