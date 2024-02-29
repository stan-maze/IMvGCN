"""
@Project   : IMvGCN
@Time      : 2021/10/4
@Author    : Zhihao Wu
@File      : main.py
"""

import os
import warnings
import random
import torch
import configparser
import numpy as np
from args import parameter_parser
from utils import tab_printer
from train import train
import datetime


def write_to_file(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as file:  # "a" 指示以追加模式打开文件
        file.write(f"{timestamp}: \n{message}\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = parameter_parser()
    device = torch.device("cpu" if args.device == "cpu" else "cuda:" + args.device)

    config = configparser.ConfigParser()
    config_path = "./config.ini"
    config.read(config_path)
    # for section in config.sections():
    #     print(f"[{section}]")
    #     for key, value in config.items(section):
    #         print(f"{key} = {value}")
    # print()
    args.lr = config.getfloat(args.dataset, "lr")
    args.num_epoch = config.getint(args.dataset, "epoch")
    args.alpha = config.getfloat(args.dataset, "alpha")
    args.Lambda = config.getfloat(args.dataset, "Lambda")
    args.dim1 = config.getint(args.dataset, "dim1")
    args.dim2 = config.getint(args.dataset, "dim2")

    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)
    # exit(0)

    all_ACC = []
    all_F1 = []
    all_TIME = []

    for i in range(args.n_repeated):
        ACC, F1, Time = train(args, device)
        all_ACC.append(ACC)
        all_F1.append(F1)
        all_TIME.append(Time)

    txt = (
        f"{args.dataset}\t{args.ratio}\n"
        + "ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100)
        + "F1 : {:.2f} ({:.2f})".format(np.mean(all_F1) * 100, np.std(all_F1) * 100)
        + "=" * 15
        + "\n"
    )
    write_to_file(txt)

    print("====================")
    print(args.dataset, args.ratio)
    print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    print("F1 : {:.2f} ({:.2f})".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
    print("====================")
