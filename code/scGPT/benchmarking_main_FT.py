import torch
from benchmarking_dataloader_FT import dataloader
import argparse
from tqdm import tqdm
from model import TransformerModel
from torch import optim
from tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score
import pandas as pd
from torch import nn
from peft import LoraConfig, get_peft_model
from utils import load_pretrained
import pickle
import json
from tokenizer.gene_tokenizer import GeneVocab


parser = argparse.ArgumentParser(description='AI4Bio')

# training parameters
parser.add_argument('--ep_num', type=int, default=3, help='epoch number of training')
parser.add_argument('--train_batch_size', type=int, default=256, help='')
parser.add_argument('--test_batch_size', type=int, default=256, help='')
parser.add_argument('--lr', type=int, default=0.0001, help='')
parser.add_argument('--data_path', type=str, default='/blue/qsong1/wang.qing/benchmark_dataset_API/lxndt_filter/', help='')
parser.add_argument('--train_rate', type=float, default=0.8, help='')
parser.add_argument('--ft_list', type=list, default=['out_proj'], help='')
parser.add_argument('--max_length', type=int, default=1200, help='')

args = parser.parse_args()



def get_model():
    vocab_file = "/blue/qsong1/wang.qing/benchmark_scLLM_API/scGPT-main/scGPT-main/scgpt/save/vocab.json"
    model_config_file = "/blue/qsong1/wang.qing/benchmark_scLLM_API/scGPT-main/scGPT-main/scgpt/save/args.json"
    model_file = "/blue/qsong1/wang.qing/benchmark_scLLM_API/scGPT-main/scGPT-main/scgpt/save/best_model.pt"
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=True,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    load_pretrained(model, torch.load(model_file, map_location='cuda'), verbose=False)
    model.to('cuda')
    # print(model)

    class FinetuningModel(nn.Module):
        def __init__(self, input_size=512, hidden_size=512, num_classes=2):
            super(FinetuningModel, self).__init__()
            self.scGPTmodel = model
            #print(self.UCEmodel.transformer_encoder)
            # Adding Lora : QKV
            config = LoraConfig(r=8,
                                lora_alpha=8,
                                target_modules=args.ft_list,
                                lora_dropout=0.05,
                                bias="none",
                                task_type="SEQ_CLS")  # [CAUSAL_LM,FEATURE_EXTRACTION,QUESTION_ANS,SEQ_2_SEQ_LM,SEQ_CLS,TOKEN_CLS]

            get_peft_model(self.scGPTmodel, config)

            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)


        def forward(self, input_gene_ids, expr, src_key_padding_mask):
            embedding = self.scGPTmodel._encode(
                input_gene_ids,
                expr,
                src_key_padding_mask=src_key_padding_mask
            )
            embedding = embedding[:, 0, :]
            x = self.fc1(embedding)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x, embedding

    FTmodel = FinetuningModel(input_size=512, hidden_size=512, num_classes=2)
    total_params = sum(p.numel() for p in FTmodel.parameters())
    print(f"Total number of parameters: {total_params}")

    return FTmodel

def run():
    seed = 24
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model()
    loss_function = torch.nn.BCEWithLogitsLoss()
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(model_parameters, lr=args.lr)

    model = model.to(device)

    train_data_loader, test_data_loader, vocab, model_configs= dataloader(args)

    # for f in range(args.folds):
    for epoch in range(args.ep_num):
        loss_sum = 0
        pred_all = []
        lbl_all = []

        with tqdm(train_data_loader, ncols=80, position=0, leave=True) as batches:
            for b in batches:  # sample
                input_gene_ids, expr, labels, masked_expr  = b['gene'], b['expr'], b['lbl'], b['masked_expr']
                input_gene_ids = input_gene_ids.to(device)
                expr = expr.to(device)
                src_key_padding_mask = input_gene_ids.eq(
                    vocab[model_configs["pad_token"]]
                )
                pred, ebd = model(input_gene_ids, expr, src_key_padding_mask)
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
                    input_gene_ids, expr, labels, masked_expr = b['gene'], b['expr'], b['lbl'], b['masked_expr']
                    input_gene_ids = input_gene_ids.to(device)
                    expr = expr.to(device)
                    src_key_padding_mask = input_gene_ids.eq(
                        vocab[model_configs["pad_token"]]
                    )
                    pred, ebd = model(input_gene_ids, expr, src_key_padding_mask)
                    loss = loss_function(pred, labels.to(device))

                    loss_sum = loss_sum + loss
                    pred_all.extend(pred)
                    lbl_all.extend(labels)
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
                torch.save(model.state_dict(), './output/model.pth')


if __name__ == '__main__':
    run()


