import torch
from benchmarking_dataloader_EBD import dataloader
import argparse
from tqdm import tqdm
from transformers import BertTokenizer
from torch import optim
from tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score
import pandas as pd
from torch import nn
import csv


class MLP_Classifier(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_classes=2):
        super(MLP_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

import pickle

parser = argparse.ArgumentParser(description='AI4Bio')

# training parameters
parser.add_argument('--ep_num', type=int, default=10, help='epoch number of training')
parser.add_argument('--train_batch_size', type=int, default=128, help='')
parser.add_argument('--test_batch_size', type=int, default=256, help='')
parser.add_argument('--data_path', type=str, default='/blue/qsong1/wang.qing/benchmark_dataset_API/cellPLM/embeds/', help='')
parser.add_argument('--label_path', type=str, default='/blue/qsong1/wang.qing/benchmark_dataset_API/cellPLM/labels/', help='')

parser.add_argument('--lr', type=int, default=0.0001, help='')
parser.add_argument('--train_rate', type=float, default=0.8, help='')
parser.add_argument('--ft_list', type=list, default=['none'], help='')

args = parser.parse_args()




def run():
    seed = 24
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MLP_Classifier().to(device)


    loss_function = torch.nn.BCEWithLogitsLoss()
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(model_parameters, lr=args.lr)


    train_data_loader, test_data_loader = dataloader(args)

    # for f in range(args.folds):
    for epoch in range(args.ep_num):
        loss_sum = 0
        pred_all = []
        lbl_all = []
        with tqdm(train_data_loader, ncols=80, position=0, leave=True) as batches:
            for b in batches:  # sample
                input_embeds, labels = b  # batch_size*seq_len
                pred = model(input_embeds.to(device))
                loss = loss_function(pred, labels.to(device))


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum = loss_sum + loss
                pred_all.extend(pred)
                lbl_all.extend(labels)

        pred_all_ = torch.stack(pred_all)
        lbl_all_ = torch.stack(lbl_all)
        acc = Accuracy_score(pred_all_, lbl_all_)
        f1 = F1_score(pred_all_, lbl_all_)
        try:
            aur = AUROC_score(pred_all_, lbl_all_)
        except:
            aur = 0.0
        pre = Precision_score(pred_all_, lbl_all_)
        rcl = Recall_score(pred_all_, lbl_all_)
        print('Training epoch:', epoch, 'loss:', loss_sum/len(batches), 'Accuracy:', round(acc , 4), 'AUROC:',round(aur, 4),
            'Precision:', round(pre , 4), 'Recall:', round(rcl , 4), 'F1:',round(f1 , 4))



        loss_sum = 0
        pred_all = []
        lbl_all = []
        ebd_all = []
        with torch.no_grad():
            with tqdm(test_data_loader, ncols=80, position=0, leave=True) as batches:
                for b in batches:  # sample
                    input_embeds, labels = b  # batch_size*seq_len
                    pred = model(input_embeds.to(device))
                    loss = loss_function(pred, labels.to(device))

                    loss_sum = loss_sum + loss
                    pred_all.extend(pred)
                    lbl_all.extend(labels)
                    ebd_all.extend(input_embeds.to('cpu'))

            pred_all_ = torch.stack(pred_all)
            lbl_all_ = torch.stack(lbl_all)
            ebd_all_ = torch.stack(ebd_all)
            acc = Accuracy_score(pred_all_, lbl_all_)
            f1 = F1_score(pred_all_, lbl_all_)
            try:
                aur = AUROC_score(pred_all_, lbl_all_)
            except:
                aur = 0.0
            pre = Precision_score(pred_all_, lbl_all_)
            rcl = Recall_score(pred_all_, lbl_all_)

            print('Testing epoch:', epoch, 'loss:', loss_sum/len(batches), 'Accuracy:', round(acc , 4), 'AUROC:',round(aur, 4),
                'Precision:', round(pre , 4), 'Recall:', round(rcl , 4), 'F1:',round(f1 , 4))

        if epoch == args.ep_num-1:
            roc={}
            roc['pred']=pred_all_
            roc['label']=lbl_all_
            roc['ebd']=ebd_all_
            with open('./output/testset_records.pkl', 'wb') as rocpkl:
                pickle.dump(roc, rocpkl)
            torch.save(model.state_dict(), './output/model.pth')  # 保存模型参数

if __name__ == '__main__':
    run()


