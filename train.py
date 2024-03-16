"""
@Project   : IMvGCN
@Time      : 2021/10/4
@Author    : Zhihao Wu
@File      : train.py
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_evaluation_results, criteria
from Dataloader import load_data
from model import IMvGCN

# import wandb


def train(args, device):
    feature_list, flt_list, flt_f, labels, idx_labeled, idx_unlabeled = load_data(
        args, device=device
    )
    print("Labeled sample:", len(idx_labeled))
    print(len(feature_list))
    [print(feature_per_v.shape) for feature_per_v in feature_list]
    from matplotlib import pyplot as plt

    # plt.scatter(range(len(feature_list[0][0].cpu())), feature_list[0][0].cpu())
    # plt.show()
    # print(feature_list[0])
    print(len(flt_list))
    [print(flt_per_v.shape) for flt_per_v in flt_list]
    print(flt_list[0])
    a1 = flt_list[0][0].to_dense().cpu().numpy()
    print(len(a1[a1 > 0]), sum(a1))
    print(a1)
    plt.scatter(range(len(a1[a1 > 0])), a1[a1 > 0])
    plt.show()
    exit(0)
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    layers = [args.dim1, args.dim2]
    num_view = len(feature_list)
    input_dims = []
    for i in range(num_view):  # 每个视图的input_dim可能一样
        input_dims.append(feature_list[i].shape[1])

    model = IMvGCN(input_dims, num_classes, args.dropout, layers, device).to(device)
    # negative log likelihood loss负对数似然损失，也就是交叉熵
    loss_function1 = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    begin_time = time.time()

    with tqdm(total=args.num_epoch, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            model.train()
            # 模型forward
            output, hidden_list, w_list = model(feature_list, flt_list, flt_f)
            output = F.log_softmax(
                output, dim=1
            )  # 输出的只是公共表示Z，需要softmax来得到预测向量

            optimizer.zero_grad()  # 开始优化，反向传播损失的梯度
            # 有标签样本的损失，就是交叉熵，有监督
            loss_nl = loss_function1(output[idx_labeled], labels[idx_labeled])
            # 无监督的重构损失
            loss_rl = criteria(
                num_view, output, w_list, feature_list, flt_list, args.Lambda
            )
            total_loss = loss_nl + args.alpha * loss_rl
            # TODO 反向传播，怎么解决W的正交性？还是说在forward里保持了正交在这里也就是正交了？
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()  # evaluate模式，dropout为0，也不保存梯度
                output, _, _ = model(feature_list, flt_list, flt_f)
                pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
                # TODO 数据集并不是半监督的，只用少部分的标签传播信息(idx_labeled)，也就是半监督问题了
                ACC, F1 = get_evaluation_results(
                    labels.cpu().detach().numpy()[idx_unlabeled],
                    pred_labels[idx_unlabeled],
                )
                pbar.set_postfix(
                    {
                        "Loss": "{:.6f}".format(total_loss.item()),
                        "ACC": "{:.2f}".format(ACC * 100),
                        "F1": "{:.2f}".format(F1 * 100),
                    }
                )
                pbar.update(1)

            wandb.log({"accuracy": ACC, "F1": F1, "loss": total_loss.item()})
            del output, hidden_list, w_list
            torch.cuda.empty_cache()

    cost_time = time.time() - begin_time
    model.eval()
    output, _, _ = model(feature_list, flt_list, flt_f)
    print("Evaluating the model")
    pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
    ACC, F1 = get_evaluation_results(
        labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled]
    )
    print("ACC: {:.2f}, F1: {:.2f}".format(ACC * 100, F1 * 100))

    return ACC, F1, cost_time
