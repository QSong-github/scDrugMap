from sklearn.model_selection import KFold

from data_collator_FT import DataCollator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset, random_split
import pandas as pd
import torch
import anndata
import pickle

import os
import numpy as np
from tokenizer import GeneVocab
import json


class BioDataset(torch.utils.data.Dataset):
    def __init__(self, args, dt_path,vocab,model_configs):
        print('processing '+ dt_path)
        self.vocab = vocab
        self.model_configs = model_configs
        self.max_length = args.max_length


        csv_file_path = dt_path
        df = pd.read_csv(csv_file_path, sep='\t', index_col=0,
                         usecols=lambda column: column != 'Condition' and column != 'Cell_barcode')
        lbl = list(pd.read_csv(csv_file_path, sep='\t', usecols=['Condition'])['Condition'])
        label = [1 if element == 'sensitive' else 0 for element in lbl]

        adata = anndata.AnnData(df)
        adata.X = adata.X.astype(np.float32)

        # verify gene col
        adata.var["index"] = adata.var.index
        adata.var["id_in_vocab"] = [
            self.vocab[gene] if gene in self.vocab else -1 for gene in adata.var["index"]
        ]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        genes = adata.var["index"].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)
        count_matrix = adata.X

        self.count_matrix = (count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A)
        self.gene_ids = gene_ids
        self.labels = np.eye(2)[label]


    def __len__(self):
        return len(self.count_matrix)

    def __getitem__(self, idx):
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]
        # append <cls> token at the beginning
        genes = np.insert(genes, 0, self.vocab["<cls>"])
        values = np.insert(values, 0, self.model_configs["pad_value"])
        genes = torch.from_numpy(genes[:self.max_length]).long()
        values = torch.from_numpy(values[:self.max_length]).float()
        output = {
            "id": idx,
            "genes": genes,
            "expressions": values,
            "labels": self.labels[idx]
        }

        return output



def buildDataset(args):
    vocab_file = "/blue/qsong1/wang.qing/benchmark_scLLM_API/scGPT-main/scGPT-main/scgpt/save/vocab.json"
    model_config_file = "/blue/qsong1/wang.qing/benchmark_scLLM_API/scGPT-main/scGPT-main/scgpt/save/args.json"
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    dt_all = []
    files_list_all = os.listdir(args.data_path)
    for f in files_list_all:
        dt_path = args.data_path + f
        dataset = BioDataset(args,dt_path,vocab,model_configs)
        dt_all.append(dataset)

    # 使用ConcatDataset来合并它们
    merged_dataset = ConcatDataset(dt_all)
    train_size = int(args.train_rate * len(merged_dataset))
    test_size = len(merged_dataset) - train_size

    train_dataset, test_dataset = random_split(merged_dataset, [train_size, test_size])



    return train_dataset, test_dataset,vocab,model_configs




def dataloader(args):
    train_dataset, test_dataset,vocab,model_configs = buildDataset(args)
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[model_configs["pad_token"]],
        pad_value=model_configs["pad_value"],
        do_mlm=False,
        do_binning=True,
        max_length=1200,
        sampling=True,
        keep_first_n_tokens=1,
    )
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=collator,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), args.train_batch_size),
        pin_memory=True,
        shuffle=True
    )
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.train_batch_size,
        collate_fn=collator,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), args.train_batch_size),
        pin_memory=True,
        shuffle=False
    )

    return train_data_loader,test_data_loader,vocab,model_configs






