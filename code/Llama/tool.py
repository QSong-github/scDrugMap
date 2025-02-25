from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import torch

def Accuracy_score(pred, labels):
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    acc = accuracy_score(max_prob_index_labels, max_prob_index_pred)

    return acc

def F1_score(pred, labels):
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    F1 = f1_score(max_prob_index_pred, max_prob_index_labels)

    return F1

def AUROC_score(pred, labels):
    max_prob_index_pred = pred[:, 1].view(-1, 1).cpu().detach().numpy()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    AUROC = roc_auc_score(max_prob_index_labels, max_prob_index_pred)

    return AUROC

def Precision_score(pred, labels):
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    pre = precision_score(max_prob_index_labels, max_prob_index_pred,zero_division=1)

    return pre

def Recall_score(pred, labels):
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    rcl = recall_score(max_prob_index_labels, max_prob_index_pred,zero_division=1)

    return rcl


from sklearn.metrics import average_precision_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score

def AP_score(pred, labels):
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    aps = average_precision_score(max_prob_index_labels, max_prob_index_pred)

    return aps

def AMI(pred, labels):
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    ami = adjusted_mutual_info_score(max_prob_index_labels, max_prob_index_pred)

    return ami

def ARI(pred, labels):
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    ari = adjusted_rand_score(max_prob_index_labels, max_prob_index_pred)

    return ari
