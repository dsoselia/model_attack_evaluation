### Imports ###
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import sys
import os

# Local Imports
from model_src import WideResNet
from attacks import *
import params


### Functions ###

class MingdDataset(torch.utils.data.Dataset):
    def __init__(self, model, transform=None, load_data=None):
        self.model = model
        if self.model:
            model.eval()
        self.transform = transform
        self.shape = (3, 32, 32)
        self.data = torch.tensor([])
        self.targets = torch.tensor([])
        if load_data:
            self.load_dataset(load_data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform:
            data = self.transform(data)
        return data, self.targets[index].long()

    def __len__(self):
        return self.data.shape[0]

    def gen_data_batch(self, num, lr, args, verbose=False, batch=None):
        if batch:
            new_data, new_targets = batch[0].to(args.device), batch[1].to(args.device)
            new_data.requires_grad = True
        else:
            new_data = torch.rand(num, *self.shape, requires_grad=True, device=args.device)
            new_targets = torch.randint(0, args.num_classes, (num,), device=args.device)

        if verbose:
            print(torch.sum(self.model(new_data).max(1)[1] == new_targets).item(), "out of", num)
        for i in tqdm(range(args.num_iter), file=sys.stdout):
            preds = self.model(new_data)
            loss = -1 * nn.CrossEntropyLoss()(preds, new_targets)
            loss.backward()
            grads = new_data.grad.detach()
            if (i+1) % 100 == 0 and verbose:
                print()
                print(np.percentile(grads.cpu().detach(), list(range(0, 101, 25))))
                print(torch.sum(preds.max(1)[1] == new_targets).item(), "out of", num)
            new_data.data += lr * (grads / norms_l2(grads + 1e-12))
            new_data.data = torch.clamp(new_data.detach(), 0, 1)
            new_data.grad.zero_()
        self.data = torch.cat((self.data, new_data.cpu().detach()))
        self.targets = torch.cat((self.targets, new_targets.cpu().detach()))

    def gen_dataset(self, num, lr, args, verbose=False):
        dataloader = [None] * ((num-1) // args.batch_size + 1)
        if "random" not in args.dataset:
            transform_test = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.float())])
            train = "train" in args.dataset
            dataset_class = datasets.CIFAR100 if "CIFAR100" in args.dataset else datasets.CIFAR10
            dataset = dataset_class("../data", train=train, transform=transform_test)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        for i, batch in enumerate(dataloader):
            print("Generating batch:", i+1)
            self.gen_data_batch(args.batch_size, lr, args, verbose=verbose, batch=batch)
            if (i+1) * args.batch_size >= num:
                break

    def save_dataset(self, folder):
        with open(os.path.join(folder, 'data.pt'), 'wb') as f:
            torch.save(self.data, f)
        with open(os.path.join(folder, 'targets.pt'), 'wb') as f:
            torch.save(self.targets, f)

    def load_dataset(self, folder):
        with open(os.path.join(folder, 'data.pt'), 'rb') as f:
            self.data = torch.load(f)
        with open(os.path.join(folder, 'targets.pt'), 'rb') as f:
            self.targets = torch.load(f)


def update_args(args, cofig_dict):
    for key, value in cofig_dict.items():
        setattr(args, key, value)


### Run Main ###

if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file is not None else args

    wandb_config = vars(args)
    run = wandb.init(project="attack", config=wandb_config)
    update_args(args, dict(run.config))
    run.log({"filename": __file__})

    print(args)
    args.device = torch.device("cuda:{0}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    print(args.device)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    n_class = {"CIFAR10": 10, "CIFAR100": 100}
    args.num_classes = n_class[args.dataset]

    teacher = WideResNet(n_classes=args.num_classes, depth=28, widen_factor=10, normalize=args.normalize,
                         dropRate=0.3)
    teacher = nn.DataParallel(teacher).to(args.device)
    teacher_dir = "model_teacher_normalized" if args.normalize else "model_teacher_unnormalized"
    path = f"../models/{args.dataset}/{teacher_dir}/final"
    teacher.load_state_dict(torch.load(f"{path}.pt"))
    teacher.eval()

    args.dataset += "-train"
    dataset = MingdDataset(teacher)
    dataset.gen_dataset(args.epochs, 0.01, args, verbose=True)
    print(dataset.data.shape)
    dataset.save_dataset("../data/cust_cifar10/train/")

    args.dataset = args.dataset[:-6] + "-test"
    print(args.dataset)
    dataset = MingdDataset(teacher)
    dataset.gen_dataset(args.epochs, 0.01, args, verbose=True)
    print(dataset.data.shape)
    dataset.save_dataset("../data/cust_cifar10/test/")

    args.dataset = "random"
    dataset = MingdDataset(teacher)
    dataset.gen_dataset(args.epochs, 0.01, args, verbose=True)
    print(dataset.data.shape)
    dataset.save_dataset("../data/cust_cifar10/random/")
