import torch
from benchmarking_dataloader import KfoldDataset, dataloader
import argparse
from tqdm import tqdm
from torch import optim, nn
from tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score
import pandas as pd
import csv
from get_embeds import get_embed
class MLP_Classifier(nn.Module):
    def __init__(self, input_size=4096, hidden_size=512, num_classes=2):
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

def prepare():
    parser = argparse.ArgumentParser(description='AI4Bio')

    # training parameters
    parser.add_argument('--ep_num', type=int, default=1, help='epoch number of training')
    parser.add_argument('--train_batch_size', type=int, default=1, help='')
    parser.add_argument('--test_batch_size', type=int, default=1, help='')
    parser.add_argument('--data_path', type=str, default='/home/exouser/llama2/LLAMA/samples/', help='')
    parser.add_argument('--label_path', type=str, default='', help='')
    parser.add_argument('--records_path', type=str, default='/home/exouser/llama2/LLAMA/DRMref_datasets_malignantCells_number_details.xlsx', help='')
    parser.add_argument('--lr', type=int, default=0.0001, help='')
    parser.add_argument('--folds', type=int, default=10, help='')
    parser.add_argument('--folds_list', type=list, default=[9,8,7,6,5,4, 3, 1, 2, 0], help='')

    parser.add_argument('--task', type=str, default='Drug_type', help='')
    ########################## choice of task ##############################6
    # ['all','Tissue','Cancer_type_level1','Cancer_type_level2','Drug_type','Regimen']


    parser.add_argument('--subtask', type=str, default='Immunotherapy', help='')
    ########################## choice of subtask ##############################

    ############'Tissue'###########3
    # ['Bone marrow aspirate','Peripheral blood mononuclear cells','Cell line']

    ############'Cancer_type_level2'###########20
    # ['Refractory multiple myeloma (MM)','Prostate cancer','Neuroblastoma','Chronic lymphocytic leukemia (CLL)','Small cell lung cancer (SCLC)']
    # ['Refractory acute myeloid leukemia (AML)','Triple-negative breast cancer (TNBC)','T-cell acute lymphoblastic leukemia (ALL)']
    # ['BRAFV600E-mutant melanoma','B+E2:E57RAFV600E-mutant melanoma']

    ############'Regimen'###########29
    # ['pomalidomide + dexamethasone OR lenalidomide + dexamethasone']
    # ['DARA-KRD (daratumumab + carfilzomib + lenalidomide + dexamethasone)']
    # ['paclitaxel','TAE684']
    # ['ficlatuzumab','enzalutamide','dasatinib']
    # ['vemurafenib','vemurafenib + combimetinib','vemurafenib + trametinib']

    ############'Drug_type'###########3
    # ['Targeted therapy','Immunotherapy','Chemotherapy']

    ########################## choice of subtask ##############################


    # model parameters
    parser.add_argument('--hidden_size', type=int, default=512, help='')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='')
    parser.add_argument('--intermediate_size', type=int, default=2048, help='')
    parser.add_argument('--max_position_embeddings', type=int, default=512, help='')

    args = parser.parse_args()
    print('current task:' + args.task)
    print('current subtask:' + args.subtask)
    if args.task=='all':
        data_list = ['all']
    else:
        # classify data
        df = pd.read_excel(args.records_path)
        task_file = list(df.apply(lambda row: (row['dataset_subgroup'], row[args.task]), axis=1))
        merged_data = {}
        for entry in task_file:
            key = entry[1]
            if key not in merged_data:
                merged_data[key] = []
            merged_data[key].append(entry[0])
        # for k in merged_data:
        #     print(k)
        # print(len(merged_data))
        data_list = merged_data[args.subtask]


    print(data_list)
    trdt_list, tedt_list = KfoldDataset(args,data_list)


    return args, trdt_list, tedt_list


def run():
    seed = 24
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args, trdt_list, tedt_list = prepare()

    for f in args.folds_list:

        model = MLP_Classifier()
        # print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
        loss_function = torch.nn.BCEWithLogitsLoss()
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr)

        model = model.to(device)

        train_data_loader, test_data_loader = dataloader(current_fold=f, train_list=trdt_list, test_list=tedt_list,
                                                         tr_bs=args.train_batch_size, te_bs=args.test_batch_size)

        # for f in range(args.folds):
        for epoch in range(args.ep_num):
            loss_sum = 0
            pred_all = []
            lbl_all = []

            with tqdm(train_data_loader, ncols=80, position=0, leave=True) as batches:
                for b in batches:  # sample
                    input_embeds, labels = b  # batch_size*seq_len
                    embeds = get_embed(input_embeds.to(device))
                    pred = model(embeds)
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
            if epoch == 10:
                print('Training epoch:', epoch, 'Current_fold:', f, 'loss:', loss_sum, 'Accuracy:', acc, 'AUROC:',aur,
                  'Precision:', pre, 'Recall:', rcl, 'F1:',f1)

            loss_sum = 0
            pred_all = []
            lbl_all = []

            with torch.no_grad():
                with tqdm(test_data_loader, ncols=80, position=0, leave=True) as batches:
                    for b in batches:  # sample
                        input_embeds, labels = b  # batch_size*seq_len
                        embeds = get_embed(input_embeds.to(device))
                        pred = model(embeds)
                        loss = loss_function(pred, labels.to(device))

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
                if epoch == 0:
                    print('Testing epoch:', epoch, 'Current_fold:', f, 'loss:', loss_sum, 'Accuracy:', round(acc, 4),
                          'AUROC:', round(aur, 4),
                          'Precision:', round(pre, 4), 'Recall:', round(rcl, 4), 'F1:', round(f1, 4))
                    data = [args.task, args.subtask, f, round(acc, 4), round(aur, 4), round(pre, 4), round(rcl, 4),
                            round(f1, 4), len(pred_all_)]

                  
                    with open('output/metrics_embeds.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        
                        writer.writerow(data)
        #torch.save(model.state_dict(), './model_save/model.ckpt')


if __name__ == '__main__':
    run()


