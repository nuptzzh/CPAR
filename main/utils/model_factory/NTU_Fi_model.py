import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.myfunction import ToVariable
import numpy as np
from typing import List

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_key = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x_t, h_prev):
        """
        Input:  x_t has dimensions [B, C]
                h_prev has dimentions [B, H]

        Output: context has dimensions [B, 1]

        Where B: batch size, C: number of channel, H: hidden size
        """

        query = self.W_query(h_prev)
        keys = self.W_key(x_t)

        # Batch matrix multiplication
        energy = self.V(torch.tanh(query + keys))
        attention_weights = F.softmax(energy, dim=1)

        context = torch.sum(x_t * attention_weights, dim=1).unsqueeze(-1)
        
        return context

class AttentionLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLSTMCell, self).__init__()

        self.lstm_cell = torch.nn.LSTMCell(input_size+1, hidden_size)
        self.attention = Attention(input_size, hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        """
        Input:  x_t has dimensions [B, C]
                h_prev has dimensions [B, H]
                c_prev has dimensions [B, H]

        Output: h_t has dimensions [B, H]
                c_t has dimensions [B, H]

        Where B: batch size, C: number of channel, H: hidden size
        """

        context = self.attention(x_t, h_prev)

        lstm_input = torch.cat((x_t, context), dim=1)
        h_t, c_t = self.lstm_cell(lstm_input, (h_prev, c_prev))

        return h_t, c_t


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, output_dim, hidden_size = 64):
        super(AttentionLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = AttentionLSTMCell(self.input_size, self.hidden_size)
        self.fc = nn.LazyLinear(output_dim)

    def forward(self, x, init_states=None):
        """
        Input : x has dimensions [L, B, C]
        
        Output: h_t has dimensions [B, H]
                c_t has dimensions [B, H]
        
        (L: sequence lenght, B: batch size, C: number of channel) 
        """
        x = x.permute(2,0,1)
        L, B, _ = x.size()

        h_t, c_t = (torch.zeros(B, self.hidden_size).to(x.device),
                    torch.zeros(B, self.hidden_size).to(x.device)) if init_states is None else init_states

        for i in range(L):

            h_t, c_t = self.cell(x[i], h_t, c_t)
        outputs = self.fc(h_t)
        return h_t, outputs


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
        self.mWDN2_H = nn.Linear(int(seq_len/2),int(seq_len/2))
        self.mWDN2_L = nn.Linear(int(seq_len/2),int(seq_len/2))

        self.a_to_x = nn.AvgPool1d(2)  
        self.sigmoid = nn.Sigmoid()

        
        self.srnn_x=ShallowRNN(subcarrier_num, 64,'LSTM', [128, 128], [.0, .0])
        self.srnn_x1=ShallowRNN(subcarrier_num, 64,'LSTM', [128, 128], [.0, .0])
        self.srnn_x2=ShallowRNN(subcarrier_num, 64,'LSTM', [128, 128], [.0, .0])
        self.fc = nn.LazyLinear(activity_num)

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

        # x0 = input.permute(2,0,1)
        # xh_1 = xh_1.permute(2,0,1)
        # xl_1 = xl_1.permute(2,0,1)
        # xh_2 = xh_2.transpose(1,2)
        # xl_2 = xl_2.transpose(1,2)

        h1,x0_out= self.srnn_x(input)
        h2,xh1_out= self.srnn_x1(xh_1)
        h3,xl1_out= self.srnn_x1(xl_1)
        h4,xh2_out= self.srnn_x2(xh_2)
        h3,xl2_out= self.srnn_x2(xl_2)

        output = torch.cat((x0_out,xh1_out,xl1_out,xh2_out,xl2_out), 1) 
        cls_output = self.fc(output)
        
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
        self.mWDN2_H = nn.Linear(int(seq_len/2),int(seq_len/2))
        self.mWDN2_L = nn.Linear(int(seq_len/2),int(seq_len/2))

        self.a_to_x = nn.AvgPool1d(2)  
        self.sigmoid = nn.Sigmoid()

        self.lstm_x = nn.LSTM(subcarrier_num,64,batch_first=True)
        self.lstm_x1 = nn.LSTM(subcarrier_num,64,batch_first=True)
        self.lstm_x2 = nn.LSTM(subcarrier_num,64,batch_first=True)
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

        x0 = input.transpose(1,2)
        xh_1 = xh_1.transpose(1,2)
        xl_1 = xl_1.transpose(1,2)
        xh_2 = xh_2.transpose(1,2)
        xl_2 = xl_2.transpose(1,2)

        level0_lstm,(h0,c0) = self.lstm_x(x0)
        level1_lstm_h,(h1,c1) = self.lstm_x1(xh_1)
        level1_lstm_l,(h2,c2) = self.lstm_x1(xl_1)
        level2_lstm_h,(h3,c3) = self.lstm_x2(xh_2)
        level2_lstm_l,(h4,c4) = self.lstm_x2(xl_2)
        
        embedding_output = torch.cat((h0[-1],h1[-1],h2[-1],h3[-1],h4[-1]), 1)  #,h3[-1],h4[-1]
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


