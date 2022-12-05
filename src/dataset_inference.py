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
root_path = "../files"
params_path = "../src"
split_index = 500
root = os.path.join(root_path, dataset)


### Functions ###

def load_data(folder, train_set, normalize=None, flag=0):
    # Load data
    train = (torch.load(f"{folder}/{train_set}_{v_type}_vulnerability.pt"))
    test = (torch.load(f"{folder}/test_{v_type}_vulnerability.pt"))

    if normalize is None:
        normalize = (train.mean(dim=(0, 1)), train.std(dim=(0, 1)))

    train = (train - normalize[0]) / normalize[1]
    test = (test - normalize[0]) / normalize[1]

    train = train.reshape(train.shape[0], -1)
    test = test.reshape(test.shape[0], -1)

    if flag == 1:
        return train, torch.zeros(train.shape[0]), normalize
    elif flag == 2:
        return test, torch.ones(test.shape[0]), normalize

    data = torch.cat((train, test), dim=0).reshape((-1, 30))
    target = torch.cat((torch.zeros(train.shape[0]), torch.ones(test.shape[0])), dim=0)
    rand = torch.randperm(target.shape[0])
    data = data[rand]
    target = target[rand]

    return data, target, normalize


def train_membership_inference(data, target, epochs=1000):
    model = nn.Sequential(nn.Linear(30, 100), nn.ReLU(), nn.Linear(100, 1), nn.Sigmoid())
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with tqdm(range(epochs), file=sys.stdout) as pbar:
        for _ in pbar:
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


def get_p(outputs_train, outputs_test):
    pred_test = outputs_test[:, 0].detach().cpu().numpy()
    pred_train = outputs_train[:, 0].detach().cpu().numpy()
    tval, pval = ttest_ind(pred_test, pred_train, alternative="greater", equal_var=False)
    if pval < 0:
        raise Exception(f"p-value={pval}")
    return pval


def print_inference(outputs_train, outputs_test):
    m1, m2 = outputs_test[:, 0].mean(), outputs_train[:, 0].mean()
    pval = get_p(outputs_train, outputs_test)
    print(f"p-value = {pval} \t| Mean difference = {m1 - m2}")


def eval_modes(modes, dataset_name):
    data, y, normalize = load_data(f"{root}/model_{modes[0]}_normalized", dataset_name)
    mem_infer_model = train_membership_inference(data, y)

    for mode in modes:
        print(mode)
        data, y, _ = load_data(f"{root}/model_{mode}_normalized", dataset_name, normalize=normalize)
        outputs = mem_infer_model(data)
        print_inference(outputs[y == 0], outputs[y == 1])


### Run Main ###

if __name__ == "__main__":
    modes = ['teacher', 'independent', 'distillation', 'pre-act-18']
    dist_data = ['train', 'cust_train', 'cust_test', 'cust_random']

    for dataset in dist_data:
        print(f"\nTraining on test data and {dataset} data:")
        eval_modes(modes, dataset)
