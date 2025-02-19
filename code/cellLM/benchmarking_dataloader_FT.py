from open_biomed.datasets.ctc_dataset import Zheng68k
from torch.utils.data import Dataset, DataLoader, random_split
import re
import torch
import json
import pickle

import os


def buildDataset(args):
    config = json.load(open("./configs/ctc/cellLM.json", "r"))


    dt_all = []
    files_list_all = os.listdir(args.data_path)
    for f in files_list_all:
        dt_path = args.data_path + f
        dataset = Zheng68k(path=dt_path,config=config["data"], seed=2023)
        dt_all.append(dataset)

    merged_dataset = torch.utils.data.ConcatDataset(dt_all)

    train_size = int(args.train_rate * len(merged_dataset))
    test_size = len(merged_dataset) - train_size

    train_dataset, test_dataset = random_split(merged_dataset, [train_size, test_size])



    return train_dataset, test_dataset




def dataloader(args):
    train_dataset, test_dataset = buildDataset(args)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=True)

    return train_data_loader,test_data_loader






