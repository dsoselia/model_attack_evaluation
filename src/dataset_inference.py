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
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Parameters
dataset = "CIFAR10"
v_type = "mingd"
root_path = "../files"
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


def train_membership_inference(data, target, epochs=500):
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


def get_p_values(num_ex, train, test, k):
    total = train.shape[0]
    p_values = []
    for i in range(k):
        positions = torch.randperm(total)[:num_ex]
        p_val = get_p(train[positions], test[positions])
        p_values.append(p_val)
    return p_values


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
    return mem_infer_model


def plot_eval(modes, dataset_name, min_num=5, max_num=90, mem_infer_model=None, k=100):
    data, y, normalize = load_data(f"{root}/model_{modes[0]}_normalized", dataset_name)
    if mem_infer_model is None:
        mem_infer_model = train_membership_inference(data, y)

    plt.rcParams["figure.figsize"] = (9, 5)
    plt.plot([min_num, max_num], [0.05, 0.05], 'k--', [min_num, max_num], [0.01, 0.01], 'k--')

    for mode in modes:
        print(mode)
        plot_data = []
        data, y, _ = load_data(f"{root}/model_{mode}_normalized", dataset_name, normalize=normalize)
        outputs = mem_infer_model(data)
        train_data, test_data = outputs[y == 0], outputs[y == 1]
        for num_ex in tqdm(range(min_num, max_num+1), file=sys.stdout):
            plot_data.append(get_p_values(num_ex, train_data, test_data, k))

        means = np.mean(plot_data, axis=1)
        std = np.std(plot_data, axis=1) / 2
        plt.plot(list(range(min_num, max_num+1)), means, label=mode)
        # plt.fill_between(list(range(min_num, max_num+1)), np.clip(means - std, 0, 1), np.clip(means + std, 0, 1),
        plt.fill_between(list(range(min_num, max_num + 1)), np.percentile(plot_data, 25, axis=1),
                         np.percentile(plot_data, 75, axis=1),
                         alpha=0.3)
    plt.legend()
    title_names = {'train': 'Real Train', 'cust_train': 'Adjusted Train',
                   'cust_test': 'Adjusted Test', 'cust_random': 'Adjusted Random'}
    plt.title(f"Dataset Inference Using {title_names[dataset_name]} and Real Test Data")
    plt.ylabel("Probability Model is Independent")
    plt.xlabel("Number of Samples Used")
    plt.xlim(min_num, max_num)
    plt.tight_layout()

    folder = "../plots/e_500"
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(f"{folder}/{dataset_name}_pvals.png", dpi=200)
    plt.close()


### Run Main ###

if __name__ == "__main__":
    modes = ['teacher', 'fine-tune', 'distillation', 'pre-act-18', 'extract-label', 'extract-logit', 'independent']
    dist_data = ['train', 'cust_train', 'cust_test', 'cust_random']

    for dataset in dist_data:
        print(f"\nTraining on test data and {dataset} data:")
        model = eval_modes(modes, dataset)
        plot_eval(modes, dataset, min_num=15, max_num=90, mem_infer_model=model)
