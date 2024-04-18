import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.myfunction import ToVariable
import numpy as np
from typing import List


class Conv1d_new_padding(nn.Module):
    def __init__(self, ni, nf, ks=None, stride=1, bias=False, pad_zero=True):
        super(Conv1d_new_padding, self).__init__()

        self.ks = ks
        # self.padding = nn.ConstantPad1d((0, int(self.ks-1)), 0)
        if pad_zero:
            self.padding = nn.ConstantPad1d((int((self.ks - 1) / 2), int(self.ks / 2)), 0)
        else:
            self.padding = nn.ReplicationPad1d((0, int(self.ks-1)))
        self.conv1d = nn.Conv1d(ni, nf, self.ks, stride=stride, bias=bias)

    def forward(self, x):
        out = self.padding(x)
        out = self.conv1d(out)
        return out
class ConvBlock(nn.Module):
    "Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers."
    def __init__(self, ni, nf, kernel_size=None, stride=1, act=None, pad_zero=True):
        super(ConvBlock, self).__init__()
        kernel_size = kernel_size
        self.layer_list = []

        self.conv = Conv1d_new_padding(ni, nf, ks=kernel_size, stride=stride, pad_zero=pad_zero)
        self.bn = nn.BatchNorm1d(num_features=nf)
        self.layer_list += [self.conv, self.bn]
        if act is not None: self.layer_list.append(act)

        self.net = nn.Sequential(*self.layer_list)

    def forward(self, x):
        x = self.net(x)
        return x
class Add(nn.Module):
    def forward(self, x, y):
        return x.add(y)
    def __repr__(self):
        return f'{self.__class__.__name__}'
class Squeeze(nn.Module):
    def __init__(self, dim=-1): 
        super(Squeeze, self).__init__()
        self.dim = dim
    def forward(self, x): 
        return x.squeeze(dim=self.dim)
    def __repr__(self): 
        return f'{self.__class__.__name__}(dim={self.dim})'

class _ResCNNBlock(nn.Module):
    def __init__(self, ni, nf, kss=[7, 5, 3]):
        super(_ResCNNBlock, self).__init__()
        self.convblock1 = Conv1d_new_padding(ni, nf, kss[0])
        self.convblock2 = Conv1d_new_padding(nf, nf, kss[1])
        self.convblock3 = Conv1d_new_padding(nf, nf, kss[2])

        # expand channels for the sum if necessary
        self.shortcut = ConvBlock(ni, nf, 1)
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x


class ResCNN(nn.Module):
    def __init__(self, c_in):
        super(ResCNN, self).__init__()
        nf = 64
        self.block1 = _ResCNNBlock(c_in, nf, kss=[7, 5, 3])
        self.block2 = ConvBlock(nf, nf * 2, 3)
        self.act2 = nn.LeakyReLU(negative_slope = .2)
        self.block3 = ConvBlock(nf * 2, nf * 4, 3)
        self.act3 = nn.PReLU()
        self.block4 = ConvBlock(nf * 4, nf * 2, 3)
        self.act4 = nn.ELU(alpha = .3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.act2(x)
        x = self.block3(x)
        x = self.act3(x)
        x = self.block4(x)
        x = self.act4(x)
        embedding_output = self.squeeze(self.gap(x))
        return embedding_output

class Res_CNN(nn.Module):
    def __init__(self, c_in, c_out):
        super(Res_CNN,self).__init__()
      
        nf = 64
        self.block1 = _ResCNNBlock(c_in, nf, kss=[7, 5, 3])
        self.block2 = ConvBlock(nf, nf * 2, 3)
        self.act2 = nn.LeakyReLU(negative_slope = .2)
        self.block3 = ConvBlock(nf * 2, nf * 4, 3)
        self.act3 = nn.PReLU()
        self.block4 = ConvBlock(nf * 4, nf * 2, 3)
        self.act4 = nn.ELU(alpha = .3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.lin = nn.Linear(nf * 2, c_out)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.act2(x)
        x = self.block3(x)
        x = self.act3(x)
        x = self.block4(x)
        x = self.act4(x)
        embedding_output = self.squeeze(self.gap(x))
        cls_output = self.lin(embedding_output)
        return embedding_output, cls_output

class Wavelet1_ResCNN(nn.Module):
    def __init__(self, subcarrier_num, activity_num, seq_len=300):
        super(Wavelet1_ResCNN,self).__init__()
        self.mWDN1_H = nn.Linear(seq_len,seq_len)
        self.mWDN1_L = nn.Linear(seq_len,seq_len)
        # self.mWDN2_H = nn.Linear(int(seq_len/2),int(seq_len/2))
        # self.mWDN2_L = nn.Linear(int(seq_len/2),int(seq_len/2))

        self.a_to_x = nn.AvgPool1d(2)  
        self.sigmoid = nn.Sigmoid()

        self.ResCNN_x = ResCNN(64)
        self.ResCNN_x1 = ResCNN(64)
        # self.ResCNN_x2 = ResCNN(1)
        self.output = nn.LazyLinear(activity_num)

        self.l_filter = [-0.0106,0.0329,0.0308,-0.187,-0.028,0.6309,0.7148,0.2304]
        self.h_filter = [-0.2304,0.7148,-0.6309,-0.028,0.187,0.0308,-0.0329,-0.0106]


        self.mWDN1_H.weight = torch.nn.Parameter(ToVariable(self.create_W(seq_len,False)))
        self.mWDN1_L.weight = torch.nn.Parameter(ToVariable(self.create_W(seq_len,True)))
        # self.mWDN2_H.weight = torch.nn.Parameter(ToVariable(self.create_W(int(seq_len/2),False)))
        # self.mWDN2_L.weight = torch.nn.Parameter(ToVariable(self.create_W(int(seq_len/2),True)))

    def forward(self,input):
        
        ah_1 = self.sigmoid(self.mWDN1_H(input))
        al_1 = self.sigmoid(self.mWDN1_L(input))
        xh_1 = self.a_to_x(ah_1)
        xl_1 = self.a_to_x(al_1)
        
        # ah_2 = self.sigmoid(self.mWDN2_H(xl_1))
        # al_2 = self.sigmoid(self.mWDN2_L(xl_1))
        # xh_2 = self.a_to_x(ah_2)
        # xl_2 = self.a_to_x(al_2)

        x_out = self.ResCNN_x(input)
        xh_1_out = self.ResCNN_x1(xh_1)
        xl_1_out = self.ResCNN_x1(xl_1)


        output = self.output(torch.cat((x_out,xh_1_out,xl_1_out), 1))
        #output = output.view(-1,1)
        return x_out,output     #x_out错的

    def create_W(self,P,is_l,is_comp=False):
        if is_l : 
            filter_list = self.l_filter
        else:
            filter_list = self.h_filter

        list_len = len(filter_list)

        max_epsilon = np.min(np.abs(filter_list))
        if is_comp:
            weight_np = np.zeros((P,P))
        else:
            weight_np = np.random.randn(P,P)*0.1*max_epsilon

        for i in range(0,P):
            filter_index = 0
            for j in range(i,P):
                if filter_index < len(filter_list):
                    weight_np[i][j] = filter_list[filter_index]
                    filter_index += 1
        return weight_np




class Wavelet2_ResCNN(nn.Module):
    def __init__(self, subcarrier_num, activity_num, seq_len=300):
        super(Wavelet2_ResCNN,self).__init__()
        self.mWDN1_H = nn.Linear(seq_len,seq_len)
        self.mWDN1_L = nn.Linear(seq_len,seq_len)
        self.mWDN2_H = nn.Linear(int(seq_len/2),int(seq_len/2))
        self.mWDN2_L = nn.Linear(int(seq_len/2),int(seq_len/2))

        self.a_to_x = nn.AvgPool1d(2)  
        self.sigmoid = nn.Sigmoid()

        self.ResCNN_x = ResCNN(64)
        self.ResCNN_x1 = ResCNN(64)
        self.ResCNN_x2 = ResCNN(64)
        self.output = nn.LazyLinear(activity_num)

        self.l_filter = [-0.0106,0.0329,0.0308,-0.187,-0.028,0.6309,0.7148,0.2304]
        self.h_filter = [-0.2304,0.7148,-0.6309,-0.028,0.187,0.0308,-0.0329,-0.0106]


        self.mWDN1_H.weight = torch.nn.Parameter(ToVariable(self.create_W(seq_len,False)))
        self.mWDN1_L.weight = torch.nn.Parameter(ToVariable(self.create_W(seq_len,True)))
        self.mWDN2_H.weight = torch.nn.Parameter(ToVariable(self.create_W(int(seq_len/2),False)))
        self.mWDN2_L.weight = torch.nn.Parameter(ToVariable(self.create_W(int(seq_len/2),True)))

    def forward(self,input):
        
        ah_1 = self.sigmoid(self.mWDN1_H(input))
        al_1 = self.sigmoid(self.mWDN1_L(input))
        xh_1 = self.a_to_x(ah_1)
        xl_1 = self.a_to_x(al_1)
        
        ah_2 = self.sigmoid(self.mWDN2_H(xl_1))
        al_2 = self.sigmoid(self.mWDN2_L(xl_1))
        xh_2 = self.a_to_x(ah_2)
        xl_2 = self.a_to_x(al_2)

        x_out = self.ResCNN_x(input)
        xh_1_out = self.ResCNN_x1(xh_1)
        xl_1_out = self.ResCNN_x1(xl_1)
        xh_2_out = self.ResCNN_x2(xh_2)
        xl_2_out = self.ResCNN_x2(xl_2)


        output = self.output(torch.cat((x_out, xh_1_out, xl_1_out, xh_2_out, xl_2_out), 1))
        #output = output.view(-1,1)
        return x_out,output     #x_out错的

    def create_W(self,P,is_l,is_comp=False):
        if is_l : 
            filter_list = self.l_filter
        else:
            filter_list = self.h_filter

        list_len = len(filter_list)

        max_epsilon = np.min(np.abs(filter_list))
        if is_comp:
            weight_np = np.zeros((P,P))
        else:
            weight_np = np.random.randn(P,P)*0.1*max_epsilon

        for i in range(0,P):
            filter_index = 0
            for j in range(i,P):
                if filter_index < len(filter_list):
                    weight_np[i][j] = filter_list[filter_index]
                    filter_index += 1
        return weight_np


