from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import torch

def Accuracy_score_cell(pred, labels):
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    acc = accuracy_score(max_prob_index_labels, max_prob_index_pred)

    return acc

def F1_score_cell(pred, labels):
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    F1 = f1_score(max_prob_index_pred, max_prob_index_labels)

    return F1

def AUROC_score_cell(pred, labels):
    max_prob_index_pred = pred[:, 1].view(-1, 1).cpu().detach().numpy()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    AUROC = roc_auc_score(max_prob_index_labels, max_prob_index_pred)

    return AUROC

def Precision_score_cell(pred, labels):
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    pre = precision_score(max_prob_index_labels, max_prob_index_pred, zero_division=1)

    return pre

def Recall_score_cell(pred, labels):
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    rcl = recall_score(max_prob_index_labels, max_prob_index_pred)

    return rcl

def Accuracy_score_gene(pred, labels):
    max_prob_index_pred = pred
    max_prob_index_labels = labels
    acc = accuracy_score(max_prob_index_labels, max_prob_index_pred)

    return acc

def F1_score_gene(pred, labels):
    max_prob_index_pred = pred
    max_prob_index_labels = labels
    F1 = f1_score(max_prob_index_pred, max_prob_index_labels, average='macro')

    return F1

def AUROC_score_gene(pred, labels):
    max_prob_index_pred = pred
    max_prob_index_labels = labels
    AUROC = roc_auc_score(max_prob_index_labels, max_prob_index_pred)

    return AUROC

def Precision_score_gene(pred, labels):
    max_prob_index_pred = pred
    max_prob_index_labels = labels
    pre = precision_score(max_prob_index_labels, max_prob_index_pred, average='macro',zero_division=1)

    return pre

def Recall_score_gene(pred, labels):
    max_prob_index_pred = pred
    max_prob_index_labels = labels
    rcl = recall_score(max_prob_index_labels, max_prob_index_pred, average='macro', zero_division=1)

    return rcl


def vote(logit_pair):
    a, b = logit_pair
    if a > b:
        return 0
    elif b > a:
        return 1
    elif a == b:
        return "tie"