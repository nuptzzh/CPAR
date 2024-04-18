import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.myfunction import ToVariable
import numpy as np

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

class Wavelet_LSTM(nn.Module):
    def __init__(self, subcarrier_num, activity_num, seq_len=300):
        super(Wavelet_LSTM,self).__init__()
        self.mWDN1_H = nn.Linear(seq_len,seq_len)                                                          #64,64,300  length300
        self.mWDN1_L = nn.Linear(seq_len,seq_len)
        self.mWDN2_H = nn.Linear(int(seq_len/2),int(seq_len/2))
        self.mWDN2_L = nn.Linear(int(seq_len/2),int(seq_len/2))
        self.a_to_x = nn.AvgPool1d(2)  
        self.sigmoid = nn.Sigmoid()
        self.lstm_xh1 = nn.LSTM(subcarrier_num,64,batch_first=True)
        self.lstm_xh2 = nn.LSTM(subcarrier_num,64,batch_first=True)
        self.lstm_xl2 = nn.LSTM(subcarrier_num,64,batch_first=True)
        self.output = nn.LazyLinear(activity_num)

        self.l_filter = [-0.0106,0.0329,0.0308,-0.187,-0.028,0.6309,0.7148,0.2304]
        self.h_filter = [-0.2304,0.7148,-0.6309,-0.028,0.187,0.0308,-0.0329,-0.0106]

        self.cmp_mWDN1_H = ToVariable(self.create_W(seq_len,False,is_comp=True))
        self.cmp_mWDN1_L = ToVariable(self.create_W(seq_len,True,is_comp=True))
        self.cmp_mWDN2_H = ToVariable(self.create_W(int(seq_len/2),False,is_comp=True))
        self.cmp_mWDN2_L = ToVariable(self.create_W(int(seq_len/2),True,is_comp=True))

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

        xh_1 = xh_1.transpose(1,2)
        xh_2 = xh_2.transpose(1,2)
        xl_2 = xl_2.transpose(1,2)

        level1_lstm,(h1,c1) = self.lstm_xh1(xh_1)
        level2_lstm_h,(h2,c2) = self.lstm_xh2(xh_2)
        level2_lstm_l,(h3,c3) = self.lstm_xl2(xl_2)
        embedding_output = torch.cat((h1[-1],h2[-1],h3[-1]), 1)
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

class CSI_BiLSTM(nn.Module):
    def __init__(self, subcarrier_num, activity_num):
        super(CSI_BiLSTM,self).__init__()
        self.lstm = nn.LSTM(subcarrier_num,64,num_layers=1,bidirectional=True)
        self.fc = nn.Linear(64,activity_num)
    def forward(self,x):
        x = x.permute(2,0,1)
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return ht, outputs


class CSI_GRU(nn.Module):
    def __init__(self, subcarrier_num, activity_num):
        super(CSI_GRU,self).__init__()
        self.gru = nn.GRU(subcarrier_num,64,num_layers=1)
        self.fc = nn.Linear(64,activity_num)
    def forward(self,x):
        x = x.permute(2,0,1)
        _, ht = self.gru(x)
        outputs = self.fc(ht[-1])
        return ht, outputs

class CSI_CNN_GRU(nn.Module):
    def __init__(self,subcarrier_num,activity_num):
        super(CSI_CNN_GRU,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(subcarrier_num,16,12,6),#B*16*48
            nn.ReLU(),
            nn.MaxPool1d(2),#B*16*24
            nn.Conv1d(16,32,7,3),#B*32*6
            nn.ReLU(),
        )
        self.gru = nn.GRU(6,128,num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128,activity_num),
        )
    def forward(self,x):#64*300
        x = self.encoder(x)
        x = x.permute(1,0,2)
        _, ht = self.gru(x)
        outputs = self.classifier(ht[-1])
        return ht,outputs