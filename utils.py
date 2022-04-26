# -*-coding:utf-8-*-
import os
import json
import pickle
import random
import torch
from tqdm import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

device = "cuda:0"

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)



def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def plot_confusion_matrix(y_true, y_pred, d, save_folder, title='Confusion Matrix', cmap=plt.cm.ocean):
    labels = d.keys()
    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(y_true, y_pred)  
    np.set_printoptions(precision=2)

    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]  
    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')

    title = 'Normalized confusion matrix'
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=45)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(save_folder, 'confusion_matrix.png'), format='png')
    #plt.show()


import sklearn.metrics

"""
Python compute equal error rate (eer)
ONLY tested on binary classification

:param label: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
:param pred: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
:param positive_label: the class that is viewed as positive class when computing EER
:return: equal error rate (EER)
"""


def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


from sklearn.metrics import roc_curve
from scipy.stats import norm


def plot_DET_curve(x, y, eer, save_folder):
    # ���ÿ̶ȷ�Χ
    pmiss_min = 0.001

    pmiss_max = 0.6

    pfa_min = 0.001

    pfa_max = 0.6

    # �̶�����
    pticks = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005,
              0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
              0.1, 0.2, 0.4, 0.6, 0.8, 0.9,
              0.95, 0.98, 0.99, 0.995, 0.998, 0.999,
              0.9995, 0.9998, 0.9999, 0.99995, 0.99998, 0.99999]

    # �̶�*100
    xlabels = [' 0.001', ' 0.002', ' 0.005', ' 0.01 ', ' 0.02 ', ' 0.05 ',
               '  0.1 ', '  0.2 ', ' 0.5  ', '  1   ', '  2   ', '  5   ',
               '  10  ', '  20  ', '  40  ', '  60  ', '  80  ', '  90  ',
               '  95  ', '  98  ', '  99  ', ' 99.5 ', ' 99.8 ', ' 99.9 ',
               ' 99.95', ' 99.98', ' 99.99', '99.995', '99.998', '99.999']

    ylabels = xlabels

    # ȷ���̶ȷ�Χ
    n = len(pticks)
    # ����
    for k, v in enumerate(pticks[::-1]):
        if pmiss_min <= v:
            tmin_miss = n - k - 1  # �ƶ���Сֵ����λ��
        if pfa_min <= v:
            tmin_fa = n - k - 1  # �ƶ���Сֵ����λ��
    # ����
    for k, v in enumerate(pticks):
        if pmiss_max >= v:
            tmax_miss = k + 1  # �ƶ����ֵ����λ��
        if pfa_max >= v:
            tmax_fa = k + 1  # �ƶ����ֵ����λ��

    # FRR
    plt.figure()
    plt.xlim(norm.ppf(pfa_min), norm.ppf(pfa_max))

    plt.xticks(norm.ppf(pticks[tmin_fa:tmax_fa]), xlabels[tmin_fa:tmax_fa])
    plt.xlabel('False Alarm probability (in %)')

    # FAR
    plt.ylim(norm.ppf(pmiss_min), norm.ppf(pmiss_max))
    plt.yticks(norm.ppf(pticks[tmin_miss:tmax_miss]), ylabels[tmin_miss:tmax_miss])
    plt.ylabel('Miss probability (in %)')
    plt.title('Detection error trade-off plot of CM')
    plt.plot(x, y, label='CM system ( EER = {:.3f})'.format(eer))
    plt.legend(loc='best')
    plt.savefig(os.path.join(save_folder, 'EER.png'), format='png')

    return plt


# EER
def compute_EER(frr, far):
    threshold_index = np.argmin(abs(frr - far))  
    eer = (frr[threshold_index] + far[threshold_index]) / 2
    print("eer=", eer)
    return eer


# minDCF P_miss = frr  P_fa = far
def compute_minDCF2(P_miss, P_fa):
    C_miss = C_fa = 1
    P_true = 0.01
    P_false = 1 - P_true

    npts = len(P_miss)
    if npts != len(P_fa):
        print("error,size of Pmiss is not euqal to pfa")

    DCF = C_miss * P_miss * P_true + C_fa * P_fa * P_false

    min_DCF = min(DCF)

    print("min_DCF_2=", min_DCF)

    return min_DCF


# minDCF P_miss = frr  P_fa = far
def compute_minDCF3(P_miss, P_fa, min_DCF_2):
    C_miss = C_fa = 1
    P_true = 0.001
    P_false = 1 - P_true

    npts = len(P_miss)
    if npts != len(P_fa):
        print("error,size of Pmiss is not euqal to pfa")

    DCF = C_miss * P_miss * P_true + C_fa * P_fa * P_false

    min_DCF = 1
    for dcf in DCF:
        if dcf > min_DCF_2 + 0.1 and dcf < min_DCF:
            min_DCF = dcf

    print("min_DCF_3=", min_DCF)
    return min_DCF

def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_function, logger):
    model.train()
    loss_function = loss_function
    accu_loss = torch.zeros(1).to(device) 
    accu_num = torch.zeros(1).to(device) 
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        # grad cliping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100, norm_type=2)
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            logger.info('WARNING: non-finite loss, ending training ', loss)
            # import sys
            # sys.exit(1)
            return None, None

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, loss_function, ema, logger):
    loss_function = loss_function

    model.eval()

    accu_num = torch.zeros(1).to(device)  
    accu_loss = torch.zeros(1).to(device) 

    sample_num = 0
    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        if ema is None:
            pred = model(images.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(pred, labels.long().to(device))
        else:
            # Validation: with EMA
            # the .average_parameters() context manager
            # (1) saves original parameters before replacing with EMA version
            # (2) copies EMA parameters to model
            # (3) after exiting the `with`, restore original parameters to resume training later
            with ema.average_parameters():
                pred = model(images.to(device))
                pred_classes = torch.max(pred, dim=1)[1]
                accu_num += torch.eq(pred_classes, labels.to(device)).sum()

                loss = loss_function(pred, labels.long().to(device))
        
        if not torch.isfinite(loss):
            logger.info('WARNING: non-finite loss, ending validation ', loss)
            # import sys
            # sys.exit(1)
            return None, None

        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                            accu_loss.item() / (step + 1),
                                                                            accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
