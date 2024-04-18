import torch
import torch.nn.functional as F
import numpy as np
import random
import scipy.io as scio
from .model_factory import NTU_Fi_model


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def LoadDataset(person_index, subcarrier_num, data_type, data_len, ratio, train_state, seed):
    setup_seed(seed)
    if data_type == "amp":
        x_mix = np.zeros((1, subcarrier_num, data_len)) 

    
    y_mix = np.zeros((1,))
    y_domain_mix = np.zeros((1,))

    for i in person_index:
        x = np.load(f'../Dataset/{data_type}/x_len={data_len}_subcarrier_num={subcarrier_num}_index={i}.npy')
        x = np.squeeze(x)
        x_mix = np.concatenate((x_mix,x),axis=0)

        y = np.load(f'../Dataset/{data_type}/y_len={data_len}_subcarrier_num={subcarrier_num}_index={i}.npy')
        y = np.squeeze(np.argmax(y, axis = 1))
        y_mix = np.concatenate((y_mix,y),axis=0)

        y_domain = i*np.ones(y.shape[0],)
        y_domain = np.squeeze(y_domain)
        y_domain_mix = np.concatenate((y_domain_mix,y_domain),axis=0)

    if train_state:
        x_ = x_mix[1:,]
        y_ = y_mix[1:,]
        y_domain_ = y_domain_mix[1:,]
        index = np.random.permutation([i for i in range(x_.shape[0])])
        x_ = x_[index, ...]
        y_ = y_[index, ...]
        y_domain_ = y_domain_[index, ...]

        x_train = x_[:int(ratio*x_.shape[0]), ...]
        y_train = y_[:int(ratio*x_.shape[0]), ...]
        y_domain_train = y_domain_[:int(ratio*x_.shape[0]), ...]

        x_val = x_[int(ratio*x_.shape[0]):, ...]
        y_val = y_[int(ratio*x_.shape[0]):, ...]
        y_domain_val = y_domain_[int(ratio*x_.shape[0]):, ...]
        return x_train, y_train, y_domain_train, x_val, y_val, y_domain_val
    else:
        return x_mix[1:,], y_mix[1:,]


def train(model, loss, train_dataloader, optimizer, epoch):
    model.train()
    correct = 0
    all_loss = 0
    for data_nn in train_dataloader:
        data, target, target_domain = data_nn
        target = target.long()
        target_domain = target_domain.long()
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
            target_domain = target_domain.cuda()
        # print('target shape:')
        # print(target.shape)
        optimizer.zero_grad()
        embedding_output, cls_output = model(data)
        cls_output = F.log_softmax(cls_output, dim=1)
        # print(cls_output.shape)
        result_loss = loss(cls_output, target)
        result_loss.backward()

        optimizer.step()
        all_loss += result_loss.item()*data.size()[0]
        pred = cls_output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        all_loss / len(train_dataloader.dataset),
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )

def evaluate(model, loss, test_dataloader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, target_domain in test_dataloader:
            target = target.long()
            target_domain = target_domain.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                target_domain = target_domain.cuda()
            embedding_output, cls_output = model(data)
            cls_output = F.log_softmax(cls_output, dim=1)
            test_loss += loss(cls_output, target).item()*data.size()[0]
            pred = cls_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nValidation set: Loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

    return test_loss

def test(model, test_dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            embedding_output, cls_output = model(data)
            pred = cls_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_dataloader.dataset)


def train_and_evaluate(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, save_path):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train(model, loss_function, train_dataloader, optimizer, epoch)
        test_loss = evaluate(model, loss_function, val_dataloader, epoch)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not improved.")

def get_model(modelname, subcarrier_num, num_cls, data_type):
    if data_type == "amp":
        subcarrier_num = subcarrier_num

    if modelname == "LSTM":
        model = NTU_Fi_model.CSI_LSTM(subcarrier_num, num_cls)
    if modelname == "BLSTM":
        model = NTU_Fi_model.CSI_BiLSTM(subcarrier_num, num_cls)
    if modelname == "GRU":
        model = NTU_Fi_model.CSI_GRU(subcarrier_num, num_cls)
    if modelname == "CNN_GRU":
        model = NTU_Fi_model.CSI_CNN_GRU(subcarrier_num, num_cls)
    if modelname == "Wavelet_LSTM":
        model = NTU_Fi_model.Wavelet_LSTM(subcarrier_num, num_cls)
    if modelname == "Wavelet_LSTMv2":
        model = NTU_Fi_model.Wavelet_LSTMv2(subcarrier_num, num_cls)
    if modelname == "ShallowRNN":
        model = NTU_Fi_model.ShallowRNN(subcarrier_num, num_cls,'LSTM', [64, 64], [0.2, 0.2])
    if modelname == "Wavelet_ShallowRNN":
        model = NTU_Fi_model.Wavelet_ShallowRNN(subcarrier_num, num_cls)
    if modelname == "AttentionLSTM":
        model = NTU_Fi_model.AttentionLSTM(subcarrier_num, num_cls)
    if modelname == "Wavelet1_ResCNN":
        model = NTU_Fi_model.Wavelet1_ResCNN(subcarrier_num, num_cls)
    if modelname == "Wavelet2_ResCNN":
        model = NTU_Fi_model.Wavelet2_ResCNN(subcarrier_num, num_cls)
    if modelname == "Res_CNN":
        model = NTU_Fi_model.Res_CNN(subcarrier_num, num_cls)
    return model