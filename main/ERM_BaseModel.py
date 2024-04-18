import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from utils.train_tools import *
from torchsummary import summary
import torch.nn.functional as F
import numpy as np
import random
from thop import profile,clever_format

def main(modelname, mode, batch_size, train_ratio, epochs, lr, wd, seed, person_num, activity_num, data_type, data_len, subcarrier_num, target_domain):
    setup_seed(seed)
    source_domain = [index for index in range(1, person_num + 1) if index != target_domain]
    target_domain = [target_domain]
    model_weight_path = f'model/{modelname, data_type, data_len, subcarrier_num}_{target_domain}_{epochs, batch_size, lr}.pth'

    if mode == 'only_train' or mode == 'train_test':
        x_train, y_train, y_domain_train, x_val, y_val, y_domain_val = LoadDataset(source_domain, subcarrier_num, data_type, data_len, train_ratio, True, seed)
        train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(y_domain_train))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val), torch.Tensor(y_domain_val))
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        model = get_model(modelname, subcarrier_num, activity_num, data_type).cuda()

        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = wd)
        loss = nn.NLLLoss().cuda()

        train_and_evaluate(model,
            loss_function=loss,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optim,
            epochs=epochs,
            save_path=model_weight_path)

    if mode == 'only_test' or mode == 'train_test':
        x_test, y_test = LoadDataset(target_domain, subcarrier_num, data_type, data_len, train_ratio, False, seed)
        test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        model = torch.load(model_weight_path)
        test_acc = test(model,test_dataloader)
        return test_acc

if __name__ == "__main__":
    modelname = "Wavelet1_ResCNN"# Wavelet1_ResCNN, Wavelet2_ResCNN, Res_CNN
    mode = 'train_test'#train_test,only_test
    batch_size = 64
    train_ratio = 0.8
    epochs = 500
    lr = 0.001
    wd = 0
    seed = 2023
    person_num = 4
    activity_num = 7

    data_type = 'amp'
    data_len = 300
    subcarrier_num = 64
    target_domains = [1, 2, 3, 4]#1, 2, 3, 4
    for target_domain in target_domains:
        test_acc = main(modelname, 
            mode, 
            batch_size, 
            train_ratio, 
            epochs, 
            lr, 
            wd, 
            seed, 
            person_num, 
            activity_num, 
            data_type, 
            data_len, 
            subcarrier_num, 
            target_domain)
        print(test_acc)