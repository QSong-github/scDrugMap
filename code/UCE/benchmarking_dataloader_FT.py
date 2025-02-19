"""
Dataloaders

"""

import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append('../')
from typing import Dict
import torch
import numpy as np
import pickle
import torch.utils.data as data

import os
from torch.utils.data import DataLoader, random_split
class MultiDatasetSentences(data.Dataset):
    def __init__(self, args,
                 dataset_to_protein_embeddings_path="/lfs/local/0/yanay/reduced_datasets_to_pe_chrom_5120_new.torch",
                 datasets_to_chroms_path="/lfs/local/0/yanay/dataset_to_chroms_new.pkl",
                 datasets_to_starts_path="/lfs/local/0/yanay/dataset_to_starts_new.pkl",
                 dt_path='',
                 label_path='',
                 shape=[]) -> None:
        super(MultiDatasetSentences, self).__init__()
        # self.xs = {}
        self.num_cells = {}
        self.num_genes = {}
        self.shapes_dict = {'samples': shape}
        self.args = args

        num_cells, num_genes = self.shapes_dict['samples']
        # self.xs[name] = X
        self.num_cells['samples'] = num_cells
        self.num_genes['samples'] = num_genes
        self.total_num_cells = num_cells

        self.datasets = ['samples']

        # TODO: preferably not hard-coded here
        self.dataset_to_protein_embeddings = torch.load(dataset_to_protein_embeddings_path)
        self.dataset_to_protein_embeddings['samples'] = self.dataset_to_protein_embeddings.pop('10k_pbmcs_proc')
        with open(datasets_to_chroms_path, "rb") as f:
            self.dataset_to_chroms = pickle.load(f)
            self.dataset_to_chroms['samples'] = self.dataset_to_chroms.pop('10k_pbmcs_proc')
        with open(datasets_to_starts_path, "rb") as f:
            self.dataset_to_starts = pickle.load(f)
            self.dataset_to_starts['samples'] = self.dataset_to_starts.pop('10k_pbmcs_proc')


        self.cts = np.memmap(dt_path,dtype='int64', mode='r', shape=self.shapes_dict['samples'])

        lbl = np.load(label_path)['array']
        self.lbl_onehot = np.eye(2)[lbl]

    def __getitem__(self, idx):
        for dataset in sorted(self.datasets):
            if idx==0:
                idx = idx+1
            if idx < self.num_cells[dataset]:
                counts = self.cts[idx]
                counts = torch.tensor(counts).unsqueeze(0)
                weights = torch.log1p(counts)
                weights = (weights / torch.sum(weights))
                batch_sentences, mask, seq_len, cell_sentences = \
                    sample_cell_sentences(counts, weights, dataset, self.args,
                                          dataset_to_protein_embeddings=self.dataset_to_protein_embeddings,
                                          dataset_to_chroms=self.dataset_to_chroms,
                                          dataset_to_starts=self.dataset_to_starts)
                label = self.lbl_onehot[idx]
                return batch_sentences, mask, idx, seq_len, cell_sentences, label
            else:
                idx -= self.num_cells[dataset]

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


class MultiDatasetSentenceCollator(object):
    def __init__(self, args):
        self.pad_length = args.pad_length

    def __call__(self, batch):
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length))
        mask = torch.zeros((batch_size, self.pad_length))
        cell_sentences = torch.zeros((batch_size, self.pad_length))

        labels = []

        i = 0
        max_len = 0
        for bs, msk, idx, seq_len, cs, lbl in batch:
            batch_sentences[i, :] = bs
            cell_sentences[i, :] = cs
            max_len = max(max_len, seq_len)
            mask[i, :] = msk
            labels.append(torch.tensor(lbl))
            i += 1
        labels = torch.stack(labels, dim=0)
        return batch_sentences[:, :max_len], mask[:, :max_len], cell_sentences, labels


def sample_cell_sentences(counts, batch_weights, dataset, args,
                          dataset_to_protein_embeddings,
                          dataset_to_chroms,
                          dataset_to_starts):
    dataset_idxs = dataset_to_protein_embeddings[dataset]  # get the dataset specific protein embedding idxs
    cell_sentences = torch.zeros((counts.shape[0], args.pad_length))  # init the cell representation as 0s
    mask = torch.zeros((counts.shape[0], args.pad_length))  # start of masking the whole sequence
    chroms = dataset_to_chroms[dataset]  # get the dataset specific chroms for each gene
    starts = dataset_to_starts[dataset]  # get the dataset specific genomic start locations for each gene

    longest_seq_len = 0  # we need to keep track of this so we can subset the batch at the end
    # have not use counts
    for c, cell in enumerate(counts):
        weights = batch_weights[c].numpy()
        weights = weights / sum(weights)  # RE NORM after mask

        # randomly choose the genes that will make up the sample, weighted by expression, with replacement
        choice_idx = np.random.choice(np.arange(len(weights)),
                                      size=args.sample_size, p=weights,
                                      replace=True)
        choosen_chrom = chroms[choice_idx]  # get the sampled genes chromosomes
        # order the genes by chromosome
        chrom_sort = np.argsort(choosen_chrom)
        choice_idx = choice_idx[chrom_sort]

        # sort the genes by start
        new_chrom = chroms[choice_idx]
        choosen_starts = starts[choice_idx]

        ordered_choice_idx = np.full((args.pad_length),
                                     args.cls_token_idx)  # start with cls
        # i= 0 first token is CLS
        i = 1  # continue on to the rest of the sequence with left bracket being assumed.
        # Shuffle the chroms now, there's no natural order to chromosomes
        uq_chroms = np.unique(new_chrom)
        np.random.shuffle(uq_chroms)  # shuffle

        # This loop is actually just over one cell
        for chrom in uq_chroms:
            # Open Chrom token
            ordered_choice_idx[i] = int(
                chrom) + args.CHROM_TOKEN_OFFSET  # token of this chromosome # i = 1 next token is a chrom open
            i += 1
            # now sort the genes by start order within the chroms
            loc = np.where(new_chrom == chrom)[0]
            sort_by_start = np.argsort(
                choosen_starts[loc])  # start locations for this chromsome

            to_add = choice_idx[loc[sort_by_start]]
            ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]
            i += len(to_add)
            ordered_choice_idx[i] = args.chrom_token_right_idx  # add the chrom sep again
            i += 1  # add the closing token again

        longest_seq_len = max(longest_seq_len, i)
        remainder_len = (args.pad_length - i)

        cell_mask = torch.concat((torch.ones(i),
                                  # pay attention to all of these tokens, ignore the rest!
                                  torch.zeros(remainder_len)))

        mask[c, :] = cell_mask

        ordered_choice_idx[i:] = args.pad_token_idx  # the remainder of the sequence
        cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)

    cell_sentences_pe = cell_sentences.long()  # token indices

    return cell_sentences_pe, mask, longest_seq_len, cell_sentences



def dataloader(args):
    with open('dict_samples_shape.pkl', 'rb') as f:
        shape_dict = pickle.load(f)
    dt_all = []
    files_list_all = os.listdir(args.data_path)
    for dt in files_list_all:
        sd = shape_dict[dt]
        dt_path = args.data_path + dt
        label_path = args.label_path + dt[:-11] + 'labels.npz'

        dataset = MultiDatasetSentences(
            args=args,
            dataset_to_protein_embeddings_path=args.pe_idx_path,
            datasets_to_chroms_path=args.chroms_path,
            datasets_to_starts_path=args.starts_path,
            dt_path=dt_path,
            label_path=label_path,
            shape=sd
        )
        dt_all.append(dataset)

    dataset_all = torch.utils.data.ConcatDataset(dt_all)
    # 计算训练集和验证集的大小
    train_size = int(args.train_rate * len(dataset_all))
    test_size = len(dataset_all) - train_size

    # 使用random_split进行随机分割
    train_dataset, test_dataset = random_split(dataset_all, [train_size, test_size])
    multi_dataset_sentence_collator = MultiDatasetSentenceCollator(args)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, collate_fn=multi_dataset_sentence_collator,shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, collate_fn=multi_dataset_sentence_collator,shuffle=False)
    return train_data_loader,test_data_loader
