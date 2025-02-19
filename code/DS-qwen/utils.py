class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


from sklearn.metrics import accuracy_score
import torch


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
