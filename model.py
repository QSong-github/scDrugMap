from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import torch
from transformers import BertForTokenClassification,BertForSequenceClassification
import torch.nn as nn


pretrained_model_name = "geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/" #"geneformer-12L-30M/"

def model_load_with_Lora_and_new_dict(pretrained_model_name):
    model = BertForSequenceClassification.from_pretrained(
        "geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/",
        num_labels=3,
        output_attentions=False,
        output_hidden_states=True
    )

    # dict emb update
    base_emb_dict = torch.rand(10, 256)   # 10 new tokens
    model_state_dict = model.state_dict()
    new_dict_emb_value = torch.cat((model_state_dict['bert.embeddings.word_embeddings.weight'], base_emb_dict), dim=0)

    new_dict_emb_pattern = nn.Embedding(25436, 256, padding_idx=0)

    new_dict_emb_pattern.weight = nn.Parameter(new_dict_emb_value)

    model.bert.embeddings.word_embeddings = new_dict_emb_pattern
    # classifier update
    # new_classifier = nn.Linear(256, 2)
    # model.classifier = new_classifier

    # print(model)
    # Adding Lora : wprd_emb, QKV
    # config = LoraConfig(r=8,
    #                     lora_alpha=8,
    #                     target_modules=['word_embeddings','query', 'key', 'value'],
    #                     lora_dropout=0.05,
    #                     bias="none",
    #                     task_type="FEATURE_EXTRACTION")   # [CAUSAL_LM,FEATURE_EXTRACTION,QUESTION_ANS,SEQ_2_SEQ_LM,SEQ_CLS,TOKEN_CLS]
    #
    # model = get_peft_model(model, config)
    # print(model)
    # print(model.print_trainable_parameters())

    return model


class LoRAformer(nn.Module):
    def __init__(self, args):
        super(LoRAformer, self).__init__()
        self.args =args
        self.former = model_load_with_Lora_and_new_dict(pretrained_model_name)
        self.flatten = nn.Flatten()
        self.classfier_cell = nn.Linear(256*2048, 2)
        self.classfier_gene = nn.Linear(256, 2)

    def forward(self,seq):
        output = self.former(seq)
        hidden_states = output['hidden_states'][6]

        cell_x = self.flatten(hidden_states)
        cell_x = self.classfier_cell(cell_x)

        gene_x = self.classfier_gene(hidden_states)
        gene_x = torch.squeeze(gene_x, dim=2)

        return cell_x, gene_x





