### Imports ###

import os
import sys
import argparse, params, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from importlib import reload
from tqdm.auto import tqdm
import random
from scipy.stats import combine_pvalues, ttest_ind_from_stats, ttest_ind
from functools import reduce
from scipy.stats import hmean

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Parameters
dataset = "CIFAR10"
v_type = "mingd"
root_path = "C:/Users/hasso/Desktop/College/2022Fall/CMSC614/dataset_inference_datafree/files"
params_path = "C:/Users/hasso/Desktop/College/2022Fall/CMSC614/dataset_inference_datafree/src"
split_index = 500
root = os.path.join(root_path, dataset)


### Functions ###

def load_data(folder, normalize=None, flag=0):
    # Load data
    train = (torch.load(f"{folder}/train_{v_type}_vulnerability.pt"))
    test = (torch.load(f"{folder}/test_{v_type}_vulnerability.pt"))

    if normalize is None:
        normalize = (train.mean(dim=(0, 1)), train.std(dim=(0, 1)))

    train = (train - normalize[0]) / normalize[1]
    test = (test - normalize[0]) / normalize[1]

    train = train.reshape(train.shape[0], -1)
    test = test.reshape(test.shape[0], -1)

    data = torch.cat((train, test), dim=0).reshape((-1, 30))
    y = torch.cat((torch.zeros(train.shape[0]), torch.ones(test.shape[0])), dim=0)
    rand = torch.randperm(y.shape[0])
    data = data[rand]
    y = y[rand]

    if flag == 1:
        return train, torch.zeros(train.shape[0]), normalize
    elif flag == 2:
        return test, torch.ones(train.shape[0]), normalize
    return data, y, normalize


def train_membership_inference(data, target):
    model = nn.Sequential(nn.Linear(30, 100), nn.ReLU(), nn.Linear(100, 1), nn.Sigmoid())
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with tqdm(range(1000)) as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            preds = model(data).squeeze()
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()
            pbar.set_description('loss {}'.format(loss.item()))
    model.eval()
    return model


def eval_model(model, data, target):
    preds = (model(data).squeeze() > 0.5).int()
    print(torch.sum(preds == target).item() / len(target))


### Run Main ###

if __name__ == "__main__":
    name = "teacher"

    data, y, normalize = load_data(f"{root}/model_{name}_normalized")
    print(data.abs().mean())
    mem_inf_model = train_membership_inference(data, y)
    eval_model(mem_inf_model, data, y)

    data, y, _ = load_data(f"{root}/model_{name}_normalized", normalize=normalize, flag=2)
    print(data.abs().mean())
    eval_model(mem_inf_model, data, y)

    data, y, _ = load_data(f"{root}/model_{name}_normalized_cust/train", normalize=normalize, flag=1)
    print(data.abs().mean())
    eval_model(mem_inf_model, data, y)

    data, y, _ = load_data(f"{root}/model_{name}_normalized_cust/test", normalize=normalize, flag=1)
    print(data.abs().mean())
    eval_model(mem_inf_model, data, y)
