### Imports ###

import torch
import json
import os

# Local Imports
from mingd_dataloader import MingdDataset
from model_src import WideResNet
from train import trainer
import params
import wandb

### Functions ###

def training(args, models, run = None):
    root = f"../models/{args.dataset}"
    for mode, epochs in models.items():
        print(f"\n\n--- Training {mode} model")
        args.mode = mode
        args.epochs = epochs
        args.model_id = args.mode + ("_normalized" if args.normalize else "_unnormalized")
        args.model_dir = f"{root}/model_{args.model_id}"
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        print("Model Directory:", args.model_dir)

        lr_max_temp = args.lr_max
        if args.mode == 'fine-tune':
            args.lr_max = 0.01

        with open(f"{args.model_dir}/model_info.txt", "w") as f:
            temp_device = args.device
            args.device = None
            json.dump(args.__dict__, f, indent=2)
            args.device = temp_device

        trainer(args, run)

        if args.mode == 'fine-tune':
            args.lr_max = lr_max_temp


def dataset_generator(args):
    teacher = WideResNet(n_classes=args.num_classes, depth=28, widen_factor=10, normalize=args.normalize,
                         dropRate=0.3)
    teacher = torch.nn.DataParallel(teacher).to(args.device)
    teacher_dir = "model_teacher_normalized" if args.normalize else "model_teacher_unnormalized"
    path = f"../models/{args.dataset}/{teacher_dir}/final"
    teacher.load_state_dict(torch.load(f"{path}.pt"))
    teacher.eval()

    datasets = [args.dataset + "/train", args.dataset + "/test", args.dataset + "/random"]
    for d in datasets:
        print(f"\n\n--- Generating {d} dataset")
        args.dataset = d
        dataset = MingdDataset(teacher)
        dataset.gen_dataset(args.epochs, args.lr_max, args, verbose=True)
        dataset.save_dataset(f"../data/cust_{args.dataset}/")

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



    args.num_classes = {"CIFAR10": 10, "CIFAR100": 100}[args.dataset]

    args.device = torch.device("cuda:{0}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    print(args)

    models = {# 'teacher': 50, 'independent': 30, 'pre-act-18': 30,
              # 'distillation': 30, 'fine-tune': 5,
              'extract-label': 15, 'extract-logit': 15}
    training(args, models, run)

    args.epochs = 200
    args.lr_max = 0.01
    dataset_generator(args)
