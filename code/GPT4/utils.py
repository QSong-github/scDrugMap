




from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import torch


def Accuracy_score(pred, labels):
    acc = accuracy_score(pred, labels)

    return acc

def F1_score(pred, labels):
    F1 = f1_score(pred, labels)

    return F1

def AUROC_score(pred, labels):
    AUROC = roc_auc_score(pred, labels)

    return AUROC

def Precision_score(pred, labels):
    pre = precision_score(pred, labels,zero_division=1)

    return pre

def Recall_score(pred, labels):
    rcl = recall_score(pred, labels,zero_division=1)

    return rcl
