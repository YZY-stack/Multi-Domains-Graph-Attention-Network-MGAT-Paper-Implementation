#-*- coding: utf-8 -*-
from unicodedata import bidirectional
from model.xception import Xception
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GAT, global_mean_pool 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = GAT(in_channels=512, hidden_channels=512, num_layers=3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gnn(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)

        return F.dropout(x, p=0.2, training=self.training)


# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


# FAD Module
class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 16)
        middle_filter = Filter(size, size // 16, size // 8)
        high_filter = Filter(size, size // 8, size)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 12, 299, 299]
        return out


class RNNDecoder(nn.Module):
  def __init__(self, in_f):
    super(RNNDecoder, self).__init__()
    self.LSTM = nn.LSTM(
        input_size=in_f,
        hidden_size=512,  # 256 if bidirectional is True
        num_layers=3,
        batch_first=True,
        bidirectional=False,
    )

    self.r = nn.ReLU()
    self.d = nn.Dropout(0.3)


  def attn(self, lstm_output, h_t):
    '''add attention at the end of the lstm'''
    # lstm_output [bs, clips, hiden]  h_t[bs, hiden]
    h_t = h_t.unsqueeze(2)
    # --> attn [bs, clips, 1]
    attn_weights = torch.bmm(lstm_output, h_t)
    attention = F.softmax(attn_weights.squeeze())
    # bmm : [bs, hidden, clips] [bs, clips, 1]
    attn_out = torch.bmm(lstm_output.transpose(1, 2), attention.unsqueeze(2)) # [bs, hidden, 1]

    return attn_out.squeeze()  # [bs, hidden]

  def forward(self, x):
    self.LSTM.flatten_parameters()
    x, (hn,hc) = self.LSTM(x) # x.shape -> bs,clip,512
    x_last = x[:,-1,:] # x[:,-1,:].shape [8, 512]

    # attention
    x = self.attn(x, x_last)
    return x


class Head(torch.nn.Module):
  def __init__(self, in_f):
    super(Head, self).__init__()

    self.f = nn.Flatten()
    self.d = nn.Dropout(0.75)
    self.b1 = nn.BatchNorm1d(in_f)
    self.pool = nn.AdaptiveAvgPool2d(1)

  def forward(self, x):
    x = self.pool(x)
    x = self.f(x)
    x = self.b1(x)
    x = self.d(x)

    return x


class MGAT(nn.Module):
    '''
    This code is the implementation code for MGAT paper
    Author: yanzhiyuan1114@gmail.com
    '''
    def __init__(self, num_class, height=320, width=320):
        super(MGAT, self).__init__()
        self.num_class = num_class
        self.height = height
        self.width = width
        
        self.d = nn.Dropout(0.2)
        self.FAD_head = FAD_Head(self.width)
        self.init_xcep()
        self.init_xcep_dct()

        self.fc = nn.Linear(512, self.num_class, bias=False)
        self.dct_conv = nn.Conv2d(12, 3, 1, bias=False)

        self.bilstm = RNNDecoder(512)

        self.h1 = Head(512)
        self.h2 = Head(512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.ln = nn.LayerNorm(512)
        self.lr = nn.LeakyReLU(inplace=True)

        self.conv3x3 = nn.Conv1d(512, 512, 3)

        self.gnn = GNN()

    def get_xcep_state_dict(self, pretrained_path='pretrained/xception-b5690688.pth'):
        # load Xception
        state_dict = torch.load(pretrained_path, map_location=device)
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        return state_dict

    def init_xcep(self):
        self.xcep = Xception(self.num_class)
        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = self.get_xcep_state_dict()
        self.xcep.load_state_dict(state_dict, False)

    def init_xcep_dct(self):
        self.xcep_dct = Xception(self.num_class)
        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = self.get_xcep_state_dict()
        self.xcep_dct.load_state_dict(state_dict, False)

    def forward(self, x, extract=False):
        batch_size = x.size(0)

        # *** obtain spatial and frequency features frame by frame *** #
        cnn_embed_seq, frequency_embed_seq = [], []
        for i in range(x.size(1)):
            # spatial
            x_s = self.xcep.features(x[:,i,:,:,:])
            x_s = self.h1(x_s)
            cnn_embed_seq.append(x_s)
            # frequency
            dct_tensor = self.FAD_head(x[:,i,:,:,:])  # bs,12,224,224
            dct_image = self.dct_conv(dct_tensor)  # bs,3,224,224
            x_f = self.xcep_dct.features(dct_image)  ## bs,1024,7,7
            x_f = self.h2(x_f)
            frequency_embed_seq.append(x_f)

        # *** embedding stack *** #
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        frequency_embed_seq = torch.stack(frequency_embed_seq, dim=0).transpose_(0, 1)

        # *** temporal feature *** #
        Y_t = self.bilstm(cnn_embed_seq).view(batch_size, 1, -1)

        # *** mean value for spatial and frequency features *** #
        Y_s = torch.mean(cnn_embed_seq, dim=1).view(batch_size, 1, -1)
        Y_f = torch.mean(frequency_embed_seq, dim=1).view(batch_size, 1, -1)

        # *** gnn implementation *** #
        edge_index = torch.tensor([ [0, 0],
                                    [0, 1],
                                    [0, 2],
                                    [1, 0],
                                    [1, 1],
                                    [1, 2],
                                    [2, 0],
                                    [2, 1],
                                    [2, 2]], dtype=torch.long)
        data_list = []
        for i in range(batch_size):
            fusion_data = torch.cat((Y_s[i,:,:], Y_t[i,:,:], Y_f[i,:,:]), dim=0)
            gnn_data = Data(x=fusion_data, edge_index=edge_index.t().contiguous())
            data_list.append(gnn_data)
        gnn_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

        self.gnn.to(device)
        for i, batch in enumerate(gnn_loader):
            fusion_feature = self.gnn(batch.to(device))

        # *** residual module *** #
        fusion_feature = self.conv3x3(torch.cat((fusion_feature.view(batch_size, -1, 1), Y_f.view(batch_size, -1, 1), Y_s.view(batch_size, -1, 1)), dim=2))

        # *** Classification *** #
        p = self.fc(self.lr(self.ln(fusion_feature.view(batch_size, -1))))
        return self.d(p) if not extract else fusion_feature


# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.
