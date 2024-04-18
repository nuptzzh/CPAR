import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.myfunction import ToVariable
import numpy as np
from typing import List

class CSI_LSTM(nn.Module):
    def __init__(self, subcarrier_num, activity_num):
        super(CSI_LSTM,self).__init__()
        self.lstm = nn.LSTM(subcarrier_num,64,num_layers=1)
        self.fc = nn.Linear(64,activity_num)
    def forward(self,x):
        x = x.permute(2,0,1)       #300,64,64
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return ht, outputs

class ShallowRNN(nn.Module):
    """
    Shallow RNN:
        first layer splits the input sequence and runs several independent RNNs.
        The second layer consumes the output of the first layer using a second
        RNN, thus capturing long dependencies.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 cell_type: str,
                 hidden_dims: List[int],
                 dropouts: List[float]):
        """
        :param input_dim: feature dimension of input sequence
        :param output_dim: feature dimension of output sequence
        :param cell_type: one of LSTM or GRU
        :param hidden_dims: list of size two of hidden feature dimensions
        :param dropouts: list of size two specifying DropOut probabilities
         after the lower and upper ShaRNN layers
        """

        super(ShallowRNN, self).__init__()

        supported_cells = ['LSTM', 'GRU']
        assert cell_type in supported_cells, \
            'Only %r are supported' % supported_cells
        cls_mapping = dict(LSTM=nn.LSTM, GRU=nn.GRU)

        self.hidden_dims = hidden_dims

        rnn_type = cls_mapping[cell_type]
        self.first_layer = rnn_type(
            input_size=input_dim, hidden_size=hidden_dims[0])
        self.second_layer = rnn_type(
            input_size=hidden_dims[0], hidden_size=hidden_dims[1])

        self.first_dropout = nn.Dropout(p=dropouts[0])
        self.second_dropout = nn.Dropout(p=dropouts[1])

        self.fc = nn.LazyLinear(output_dim)
        # Default initialization of fc layer is Kaiming Uniform
        # Try Normal Distribition N(0, 1)?

    def forward(self, x: torch.Tensor, k=10):
        """
        :param x: Tensor of shape [seq length, batch size, input dimension]
        :param k: int specifying brick size/ stride in sliding window
        :return:
        """
        x = x.permute(2,0,1) 
        bricks = self.split_by_bricks(x, k)
        num_bricks, brick_size, batch_size, input_dim = bricks.shape

        bricks = bricks.permute(1, 0, 2, 3).reshape(k, -1, input_dim)
        # bricks shape: [brick size, num bricks * batch size, input dim]
        
        first, _ = self.first_layer(bricks)
        first = self.first_dropout(first)
        first = torch.squeeze(first[-1]).view(num_bricks, batch_size, -1)
        # first_out shape: [num bricks, batch size, hidden dim[0]]

        second, (ht,ct)= self.second_layer(first)
        second = self.second_dropout(second)
        second = torch.squeeze(second[-1])
        # second shape: [batch size, hidden dim[1]]

        # second, (ht,ct)= self.second_layer(first)
        

        out = self.fc(second)
        # out shape: [batch size, output dim]

        return ht,out

    @staticmethod
    def split_by_bricks(sequence: torch.Tensor, brick_size: int):
        """
        Splits an incoming sequence into bricks
        :param sequence: Tensor of shape [seq length, batch size, input dim]
        :param brick_size: int specifying brick size
        :return split_sequence: Tensor of shape
         [num bricks, brick size, batch size, feature dim]
        """

        sequence_len, batch_size, feature_dim = sequence.shape
        num_bricks = sequence_len // brick_size
        total_len = brick_size * num_bricks

        splits = torch.split(sequence[:total_len], num_bricks, dim=0)
        split_sequence = torch.stack(splits, dim=1)

        return split_sequence


class Wavelet_ShallowRNN(nn.Module):
    def __init__(self, subcarrier_num, activity_num, seq_len=300):
        super(Wavelet_ShallowRNN,self).__init__()
        self.mWDN1_H = nn.Linear(seq_len,seq_len)
        self.mWDN1_L = nn.Linear(seq_len,seq_len)
        # self.mWDN2_H = nn.Linear(int(seq_len/2),int(seq_len/2))
        # self.mWDN2_L = nn.Linear(int(seq_len/2),int(seq_len/2))

        self.a_to_x = nn.AvgPool1d(2)  
        self.sigmoid = nn.Sigmoid()

        
        self.srnn_x=ShallowRNN(subcarrier_num, 64,'LSTM', [128, 128], [.0, .0])
        self.srnn_x1=ShallowRNN(subcarrier_num, 64,'LSTM', [128, 128], [.0, .0])

        self.output = nn.LazyLinear(activity_num)

        self.l_filter = [-0.0106,0.0329,0.0308,-0.187,-0.028,0.6309,0.7148,0.2304]
        self.h_filter = [-0.2304,0.7148,-0.6309,-0.028,0.187,0.0308,-0.0329,-0.0106]

        # self.cmp_mWDN1_H = ToVariable(self.create_W(seq_len,False,is_comp=True))
        # self.cmp_mWDN1_L = ToVariable(self.create_W(seq_len,True,is_comp=True))
        # self.cmp_mWDN2_H = ToVariable(self.create_W(int(seq_len/2),False,is_comp=True))
        # self.cmp_mWDN2_L = ToVariable(self.create_W(int(seq_len/2),True,is_comp=True))

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

        x0 = input.permute(2,0,1)
        xh_1 = xh_1.permute(2,0,1)
        xl_1 = xl_1.permute(2,0,1)
        # xh_2 = xh_2.transpose(1,2)
        # xl_2 = xl_2.transpose(1,2)

        h1,x0_out= self.srnn_x(x0)
        h2,xh1_out= self.srnn_x1(xh_1)
        h3,xl1_out= self.srnn_x1(xl_1)
        
        
        output = torch.cat((x0_out,xh1_out,xl1_out), 1)  #,h3[-1],h4[-1]
        cls_output = self.output(output)
        return h1,cls_output

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


class Wavelet_LSTM(nn.Module):
    def __init__(self, subcarrier_num, activity_num, seq_len=300):
        super(Wavelet_LSTM,self).__init__()
        self.mWDN1_H = nn.Linear(seq_len,seq_len)
        self.mWDN1_L = nn.Linear(seq_len,seq_len)
        # self.mWDN2_H = nn.Linear(int(seq_len/2),int(seq_len/2))
        # self.mWDN2_L = nn.Linear(int(seq_len/2),int(seq_len/2))

        self.a_to_x = nn.AvgPool1d(2)  
        self.sigmoid = nn.Sigmoid()

        self.lstm_x = nn.LSTM(subcarrier_num,64,batch_first=True)
        self.lstm_x1 = nn.LSTM(subcarrier_num,64,batch_first=True)
        # self.lstm_x2 = nn.LSTM(subcarrier_num,64,batch_first=True)
        self.output = nn.LazyLinear(activity_num)

        self.l_filter = [-0.0106,0.0329,0.0308,-0.187,-0.028,0.6309,0.7148,0.2304]
        self.h_filter = [-0.2304,0.7148,-0.6309,-0.028,0.187,0.0308,-0.0329,-0.0106]

        # self.cmp_mWDN1_H = ToVariable(self.create_W(seq_len,False,is_comp=True))
        # self.cmp_mWDN1_L = ToVariable(self.create_W(seq_len,True,is_comp=True))
        # self.cmp_mWDN2_H = ToVariable(self.create_W(int(seq_len/2),False,is_comp=True))
        # self.cmp_mWDN2_L = ToVariable(self.create_W(int(seq_len/2),True,is_comp=True))

        self.mWDN1_H.weight = torch.nn.Parameter(ToVariable(self.create_W(seq_len,False)))
        self.mWDN1_L.weight = torch.nn.Parameter(ToVariable(self.create_W(seq_len,True)))
        self.mWDN2_H.weight = torch.nn.Parameter(ToVariable(self.create_W(int(seq_len/2),False)))
        self.mWDN2_L.weight = torch.nn.Parameter(ToVariable(self.create_W(int(seq_len/2),True)))

    def forward(self,input):

        ah_1 = self.sigmoid(self.mWDN1_H(input))
        al_1 = self.sigmoid(self.mWDN1_L(input))
        xh_1 = self.a_to_x(ah_1)
        xl_1 = self.a_to_x(al_1)
        
        # ah_2 = self.sigmoid(self.mWDN2_H(xl_1))
        # al_2 = self.sigmoid(self.mWDN2_L(xl_1))
        # xh_2 = self.a_to_x(ah_2)
        # xl_2 = self.a_to_x(al_2)

        x0 = input.transpose(1,2)
        xh_1 = xh_1.transpose(1,2)
        xl_1 = xl_1.transpose(1,2)
        # xh_2 = xh_2.transpose(1,2)
        # xl_2 = xl_2.transpose(1,2)

        level0_lstm,(h0,c0) = self.lstm_x(x0)
        level1_lstm_h,(h1,c1) = self.lstm_x1(xh_1)
        level1_lstm_l,(h2,c2) = self.lstm_x1(xl_1)
        # level2_lstm_h,(h3,c3) = self.lstm_x2(xh_2)
        # level2_lstm_l,(h4,c4) = self.lstm_x2(xl_2)
        
        embedding_output = torch.cat((h0[-1],h1[-1],h2[-1]), 1)  #,h3[-1],h4[-1]
        cls_output = self.output(embedding_output)
        return embedding_output,cls_output

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