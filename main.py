import torch
from dataloader import KfoldDataset, dataloader
import argparse
from tqdm import tqdm
from model import LoRAformer
from torch import optim
from tool import Accuracy_score_gene, F1_score_gene, AUROC_score_gene, Recall_score_gene, Precision_score_gene, vote
from tool import Accuracy_score_cell, F1_score_cell, AUROC_score_cell, Recall_score_cell, Precision_score_cell

def prepare():
    parser = argparse.ArgumentParser(description='AI4Bio')

    # training parameters
    parser.add_argument('--ep_num', type=int, default=1, help='epoch number of training')
    parser.add_argument('--train_batch_size', type=int, default=32, help='')
    parser.add_argument('--test_batch_size', type=int, default=64, help='')
    parser.add_argument('--data_path', type=str, default='./subdt', help='')
    parser.add_argument('--lr', type=int, default=0.0001, help='learning rate')
    parser.add_argument('--folds', type=int, default=5, help='')


    args = parser.parse_args()


    trdt_list, tedt_list = KfoldDataset(args.data_path, args.folds)


    return args, trdt_list, tedt_list



def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args, trdt_list, tedt_list = prepare()

    for f in [1, 0]:  # fill fold number
        model = LoRAformer(args)

        loss_function = torch.nn.CrossEntropyLoss()
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr)

        model = model.to(device)

        train_data_loader, test_data_loader = dataloader(current_fold=f, train_list=trdt_list, test_list=tedt_list,
                                                         tr_bs=args.train_batch_size, te_bs=args.test_batch_size)
        for epoch in range(args.ep_num):

            loss_sum = 0

            Acc_cell = []
            F1_cell = []
            AUROC_cell = []
            Precision_cell = []
            Recall_cell = []

            Acc_gene = []
            F1_gene = []
            AUROC_gene = []
            Precision_gene = []
            Recall_gene = []

            with tqdm(train_data_loader, ncols=80, position=0, leave=True) as batches:
                for b in batches:  # sample
                    input_ids, labels_cell, labels_gene = b  # batch_size*seq_len
                    if torch.all(torch.logical_or(torch.all(labels_cell == torch.tensor([1, 0])),
                                                  torch.all(labels_cell == torch.tensor([0, 1])))) == True:
                        continue
                    input_ids = input_ids.to(device)
                    labels_cell = labels_cell.to(device)
                    labels_gene = labels_gene.view(-1)
                    labels_gene = labels_gene.to(device)

                    pred_cell, pred_gene = model(input_ids)
                    pred_gene = pred_gene.view(-1, 2)

                    loss_cell = loss_function(pred_cell, labels_cell.float())
                    loss_gene = loss_function(pred_gene, labels_gene)
                    loss = loss_cell + 0.02*loss_gene

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_sum = loss_sum + loss

                    # cell
                    acc = Accuracy_score_cell(pred_cell, labels_cell)
                    f1 = F1_score_cell(pred_cell, labels_cell)
                    aur = AUROC_score_cell(pred_cell, labels_cell)
                    pre = Precision_score_cell(pred_cell, labels_cell)
                    rcl = Recall_score_cell(pred_cell, labels_cell)

                    Acc_cell.append(acc)
                    F1_cell.append(f1)
                    AUROC_cell.append(aur)
                    Precision_cell.append(pre)
                    Recall_cell.append(rcl)

                    # gene
                    logit_label_paired = [item for item in list(zip(pred_gene.tolist(), labels_gene.tolist())) if item[1] != -100]  # only known gene
                    y_pred = [vote(item[0]) for item in logit_label_paired]
                    y_true = [item[1] for item in logit_label_paired]

                    acc = Accuracy_score_gene(y_pred, y_true)
                    f1 = F1_score_gene(y_pred, y_true)
                    aur = AUROC_score_gene(y_pred, y_true)
                    pre = Precision_score_gene(y_pred, y_true)
                    rcl = Recall_score_gene(y_pred, y_true)

                    Acc_gene.append(acc)
                    F1_gene.append(f1)
                    AUROC_gene.append(aur)
                    Precision_gene.append(pre)
                    Recall_gene.append(rcl)


            print('Training epoch:', epoch, 'Current_fold:', f, 'loss:', loss_sum)
            print('Task type: cell_cls', 'Accuracy:', sum(Acc_cell) / len(Acc_cell), 'AUROC:', sum(AUROC_cell) / len(AUROC_cell), 'Precision:',
                  sum(Precision_cell) / len(Precision_cell), 'Recall:', sum(Recall_cell) / len(Recall_cell), 'F1:', sum(F1_cell) / len(F1_cell))
            print('Task type: gene_cls', 'Accuracy:', sum(Acc_gene) / len(Acc_gene), 'AUROC:', sum(AUROC_gene) / len(AUROC_gene), 'Precision:',
                  sum(Precision_gene) / len(Precision_gene), 'Recall:', sum(Recall_gene) / len(Recall_gene), 'F1:', sum(F1_gene) / len(F1_gene))


            loss_sum = 0

            Acc_cell = []
            F1_cell = []
            AUROC_cell = []
            Precision_cell = []
            Recall_cell = []

            Acc_gene = []
            F1_gene = []
            AUROC_gene = []
            Precision_gene = []
            Recall_gene = []

            with (torch.no_grad()):
                with tqdm(test_data_loader, ncols=80, position=0, leave=True) as batches:
                    for b in batches:  # sample
                        input_ids, labels_cell, labels_gene = b  # batch_size*seq_len
                        if torch.all(torch.logical_or(torch.all(labels_cell == torch.tensor([1, 0])),
                                                      torch.all(labels_cell == torch.tensor([0, 1])))) == True:
                            continue
                        input_ids = input_ids.to(device)
                        labels_cell = labels_cell.to(device)
                        labels_gene = labels_gene.view(-1)
                        labels_gene = labels_gene.to(device)

                        pred_cell, pred_gene = model(input_ids)
                        pred_gene = pred_gene.view(-1, 2)

                        loss_cell = loss_function(pred_cell, labels_cell.float())
                        loss_gene = loss_function(pred_gene, labels_gene)
                        loss = loss_cell + loss_gene
                        loss_sum = loss_sum + loss

                        # cell
                        acc = Accuracy_score_cell(pred_cell, labels_cell)
                        f1 = F1_score_cell(pred_cell, labels_cell)
                        aur = AUROC_score_cell(pred_cell, labels_cell)
                        pre = Precision_score_cell(pred_cell, labels_cell)
                        rcl = Recall_score_cell(pred_cell, labels_cell)

                        Acc_cell.append(acc)
                        F1_cell.append(f1)
                        AUROC_cell.append(aur)
                        Precision_cell.append(pre)
                        Recall_cell.append(rcl)

                        # gene
                        logit_label_paired = [item for item in list(zip(pred_gene.tolist(), labels_gene.tolist())) if
                                              item[1] != -100]
                        y_pred = [vote(item[0]) for item in logit_label_paired]
                        y_true = [item[1] for item in logit_label_paired]

                        acc = Accuracy_score_gene(y_pred, y_true)
                        f1 = F1_score_gene(y_pred, y_true)
                        aur = AUROC_score_gene(y_pred, y_true)
                        pre = Precision_score_gene(y_pred, y_true)
                        rcl = Recall_score_gene(y_pred, y_true)

                        Acc_gene.append(acc)
                        F1_gene.append(f1)
                        AUROC_gene.append(aur)
                        Precision_gene.append(pre)
                        Recall_gene.append(rcl)

                    print('Testing epoch:', epoch, 'Current_fold:', f, 'loss:', loss_sum)
                    print('Task type: cell_cls','Accuracy:', sum(Acc_cell) / len(Acc_cell), 'AUROC:',sum(AUROC_cell) / len(AUROC_cell),'Precision:',
                          sum(Precision_cell) / len(Precision_cell), 'Recall:', sum(Recall_cell) / len(Recall_cell),'F1:', sum(F1_cell) / len(F1_cell))
                    print('Task type: gene_cls', 'Accuracy:', sum(Acc_gene) / len(Acc_gene), 'AUROC:',sum(AUROC_gene) / len(AUROC_gene),'Precision:',
                          sum(Precision_gene) / len(Precision_gene), 'Recall:', sum(Recall_gene) / len(Recall_gene),'F1:', sum(F1_gene) / len(F1_gene))



if __name__ == '__main__':
    run()


