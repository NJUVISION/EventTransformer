import torch


def calculate_acc(pred, gt):
    return float(torch.eq(pred.argmax(dim=1), gt).sum()) / gt.shape[0]
