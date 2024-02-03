import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import WimbledonDataset
from lstm import LSTMModel, my_Cross_Loss
from parameter_get import *


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_Wimbledon_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = WimbledonDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    return train_loader, valid_loader


def train_or_eval_model(model, loss_function, dataloader, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        point_victor, features = [d.cuda() for d in data[:]] if torch.cuda.is_available() else data[:]
        point_victor = point_victor.permute(1,0)

        final_out = model(features)
        loss = loss_function(final_out, point_victor, features[-1])
        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    avg_loss = round(np.sum(losses), 4)
    return avg_loss

def train():
    cuda_availdabe = torch.cuda.is_available()
    if cuda_availdabe:
        print('Running on GPU')
    else:
        print('Running on CPU')

    # 超参数
    epochs = 20
    batch_size = 16
    hidden_dim = 32

    num_layers = 2
    lr = 0.0001
    l2 = 0.00001

    # 维度
    input_size = 38
    n_classes = 2

    model = LSTMModel(input_size, hidden_dim, num_layers, n_classes)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    loss_function = my_Cross_Loss(r=1.0)
    #loss_function = Loss(1.0, 0.5)

    train_loader, valid_loader = get_Wimbledon_loaders(valid=0.0, batch_size=batch_size, num_workers=0)

    for e in range(epochs):
        train_loss = train_or_eval_model(model, loss_function, train_loader, optimizer, True)
        valid_loss= train_or_eval_model(model, loss_function, valid_loader)

        print('epoch: {}, train_loss: {}, valid_loss: {}'. \
            format(e + 1, train_loss/batch_size, valid_loss/batch_size))

    return model


if __name__ == "__main__":
    train(0.5)