
from benchmarking_dataloader_FT import dataloader
import argparse
from tqdm import tqdm
from torch import optim
from tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score
import pandas as pd
from torch import nn
from peft import LoraConfig, get_peft_model
import pickle
import warnings
import torch
from utils import figshare_download
from modelFT import TransformerModel




warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description='AI4Bio')

# training parameters
parser.add_argument('--ep_num', type=int, default=3, help='epoch number of training')
parser.add_argument('--train_batch_size', type=int, default=80, help='')
parser.add_argument('--test_batch_size', type=int, default=80, help='')
parser.add_argument('--label_path', type=str, default='/blue/qsong1/wang.qing/benchmark_dataset_API/UCE/labels/', help='')
parser.add_argument('--data_path', type=str, default="/blue/qsong1/wang.qing/benchmark_dataset_API/UCE/samples/", help='')
parser.add_argument('--ft_list', type=list, default=['out_proj'], help='')
parser.add_argument('--train_rate', type=float, default=0.8, help='')

parser.add_argument('--lr', type=int, default=0.0001, help='')
parser.add_argument('--pe_idx_path', type=str, default='./10k_pbmcs_proc_pe_idx.torch', help='')
parser.add_argument('--chroms_path', type=str, default='./10k_pbmcs_proc_chroms.pkl', help='')
parser.add_argument('--starts_path', type=str, default='./10k_pbmcs_proc_starts.pkl', help='')
parser.add_argument('--pad_length', type=int, default=1536,help='Batch size.')
parser.add_argument('--sample_size', type=int, default=1024,help='Number of genes sampled for cell sentence')
parser.add_argument("--cls_token_idx", type=int, default=3,help="CLS token index")
parser.add_argument("--CHROM_TOKEN_OFFSET", type=int, default=143574,help="Offset index, tokens after this mark are chromosome identifiers")
parser.add_argument("--chrom_token_left_idx", type=int, default=1,help="Chrom token left index")
parser.add_argument("--chrom_token_right_idx", type=int, default=2,help="Chrom token right index")
parser.add_argument("--pad_token_idx", type=int, default=0,help="PAD token index")
parser.add_argument('--token_dim', type=int, default=5120,help='Token dimension.')
parser.add_argument('--d_hid', type=int, default=5120,help='Hidden dimension.')
parser.add_argument('--nlayers', type=int, default=4,help='Number of transformer layers.')
parser.add_argument('--output_dim', type=int, default=1280,help='Output dimension.')
parser.add_argument('--model_loc', type=str,default=None,help='Location of the model.')



args = parser.parse_args()



def get_model(args):
    #### Set up the model ####
    token_dim = args.token_dim
    emsize = 1280  # embedding dimension
    d_hid = args.d_hid  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = args.nlayers  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 20  # number of heads in nn.MultiheadAttention
    dropout = 0.05  # dropout probability
    model = TransformerModel(token_dim=token_dim, d_model=emsize, nhead=nhead,
                             d_hid=d_hid,
                             nlayers=nlayers, dropout=dropout,
                             output_dim=args.output_dim)
    if args.model_loc is None:
        args.model_loc = "./model_files/4layer_model.torch"
        figshare_download("https://figshare.com/ndownloader/files/42706576",args.model_loc)
    # intialize as empty
    empty_pe = torch.zeros(145469, 5120)
    empty_pe.requires_grad = False
    model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
    model.load_state_dict(torch.load(args.model_loc, map_location="cpu"),strict=True)
    #print(model)


    class FinetuningModel(nn.Module):
        def __init__(self, input_size=1280, hidden_size=512, num_classes=2):
            super(FinetuningModel, self).__init__()
            self.UCEmodel = model
            #print(self.UCEmodel.transformer_encoder)
            # Adding Lora : QKV
            config = LoraConfig(r=8,
                                lora_alpha=8,
                                target_modules=args.ft_list,
                                lora_dropout=0.05,
                                bias="none",
                                task_type="SEQ_CLS")  # [CAUSAL_LM,FEATURE_EXTRACTION,QUESTION_ANS,SEQ_2_SEQ_LM,SEQ_CLS,TOKEN_CLS]

            get_peft_model(self.UCEmodel.transformer_encoder, config)  #self.transformer_encoder_lora =
            # self.UCEmodel.transformer_encoder = self.transformer_encoder_lora
            # print(self.UCEmodel.transformer_encoder)
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)


        def forward(self, seq, mask):
            seq = self.UCEmodel.pe_embedding(seq.long())
            seq = nn.functional.normalize(seq,dim=2)
            _, embedding = self.UCEmodel.forward(seq, mask=mask)  # 200*1280
            x = self.fc1(embedding)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x, embedding

    FTmodel = FinetuningModel(input_size=1280, hidden_size=512, num_classes=2)
    total_params = sum(p.numel() for p in FTmodel.parameters())
    print(f"Total number of parameters: {total_params}")

    return FTmodel

def run():
    seed = 24
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(args)
    loss_function = torch.nn.BCEWithLogitsLoss()
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(model_parameters, lr=args.lr)

    model = model.to(device)

    train_data_loader, test_data_loader = dataloader(args)


    for epoch in range(args.ep_num):
        loss_sum = 0
        pred_all = []
        lbl_all = []
    
        with tqdm(train_data_loader, ncols=80, position=0, leave=True) as batches:
            for b in batches:  # sample
                batch_sentences, mask, cell_sentences, labels  = b[0], b[1], b[2], b[3]
                batch_sentences = batch_sentences.permute(1, 0).long()
                pred,ebd = model(batch_sentences.to(device), mask=mask.to(device))
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
                    batch_sentences, mask, cell_sentences, labels  = b[0], b[1], b[2], b[3]
                    batch_sentences = batch_sentences.permute(1, 0).long()
                    pred,ebd = model(batch_sentences.to(device), mask=mask.to(device))
                    loss = loss_function(pred, labels.to(device))

                    loss_sum = loss_sum + loss
                    pred_all.extend(pred.to('cpu'))
                    lbl_all.extend(labels.to('cpu'))
                    ebd_all.extend(ebd.to('cpu'))

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


