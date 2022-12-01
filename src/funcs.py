from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from attacks import *


def get_dataloaders(dataset, batch_size, normalize=False, train_shuffle=True):
    data_source = datasets.CIFAR10 if dataset == "CIFAR10" else datasets.CIFAR100

    norm_mean = {"CIFAR10": (0.49139968, 0.48215841, 0.44653091), "CIFAR100": (0.50707516, 0.48654887, 0.44091784)}
    norm_std = {"CIFAR10": (0.24703223, 0.24348513, 0.26158784), "CIFAR100": (0.26733429, 0.25643846, 0.27615047)}

    tr_normalize = transforms.Normalize(norm_mean[dataset], norm_std[dataset]) if normalize else transforms.Lambda(
        lambda x: x)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), tr_normalize,
                                          transforms.Lambda(lambda x: x.float())])
    transform_test = transforms.Compose([transforms.ToTensor(), tr_normalize, transforms.Lambda(lambda x: x.float())])

    if not train_shuffle:
        print("No training transform and shuffle")
        transform_train = transform_test

    cifar_train = data_source("../data", train=True, download=True, transform=transform_train)
    cifar_test = data_source("../data", train=False, download=True, transform=transform_test)
    train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=train_shuffle)
    test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load(model, model_name):
    try:
        model.load_state_dict(torch.load(f"{model_name}.pt"))
    except:
        dictionary = torch.load(f"{model_name}.pt")['state_dict']
        new_dict = {}
        for key in dictionary.keys():
            new_key = key[7:]
            if new_key.split(".")[0] == "sub_block1":
                continue
            new_dict[new_key] = dictionary[key]
        model.load_state_dict(new_dict)
    return model
