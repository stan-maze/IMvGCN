"""
@Project   : IMvGCN
@Time      : 2021/10/4
@Author    : Zhihao Wu
@File      : model.py
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def ortho_norm(weight):
    # 论文提出的正交归一化
    # 在对角加一个小量是为了保证AtA满秩
    wtw = torch.mm(weight.t(), weight) + 1e-4 * torch.eye(weight.shape[1]).to(weight.device)
    L = torch.linalg.cholesky(wtw)
    weight_ortho = torch.mm(weight, L.inverse().t())
    return weight_ortho


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, num_views, activation=F.tanh):
        """单层的图卷积模块

        Args:
            input_dim (_type_): _description_
            output_dim (_type_): _description_
            num_views (_type_): 应该是冗余的参数,从解耦的角度来说,真正应该传入的是flt,在最后一层可能用到其他的view
            activation (_type_, optional): _description_. Defaults to F.tanh.
        """        
        super(GraphConvSparse, self).__init__()
        # 初始化权重矩阵，这应该是代理矩阵(论文中的proxy matrix)
        self.weight = glorot_init(input_dim, output_dim)
        # ortho_weight应该是实际使用的正交权重矩阵
        self.ortho_weight = torch.zeros_like(self.weight)
        self.activation = activation
        self.num_views = num_views

    def forward(self, inputs, flt, fea_sp=False):
        x = inputs
        # 通过代理矩阵weight进行正交归一化得到ortho_weight
        self.ortho_weight = ortho_norm(self.weight)
        # self.ortho_weight = self.weight
        if fea_sp:  #sparse化，论文没有提及，看来仓库作者在尝试相关的工作
            x = torch.spmm(x, self.ortho_weight)
        else:
            x = torch.mm(x, self.ortho_weight)  # 第一步HWv做一个transform(在归一化的语境下也是一个projection)
        x = torch.spmm(flt, x)  # 第二步使用卷积核进行卷积
        if self.activation is None:
            return x
        else:
            return self.activation(x)


class FGCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_views, activation=F.tanh):
        """最后一层的GCN,不需要激活,使用综合所有视图的Ff而不是Fv

        Args:
            input_dim (_type_): _description_
            output_dim (_type_): _description_
            num_views (_type_): _description_
            activation (_type_, optional): _description_. Defaults to F.tanh.
        """        
        super(FGCN, self).__init__()
        self.weight = nn.ParameterList()
        for i in range(num_views):  # 需要把所有视图融合，因此权重是[v1_w, v2_w, ...]
            self.weight.append(glorot_init(input_dim[i], output_dim))
        self.activation = activation
        self.num_views = num_views

    def forward(self, hidden_list, flt_f):
        ortho_weight = []
        ortho_weight.append(ortho_norm(self.weight[0])) # 真正使用的正交归一化权重
        # TODO 不懂，这里的hidden_list应该相当于input，这个中心化操作是为何
        hidden_list[0] = hidden_list[0] - hidden_list[0].mean(dim=0)
        hidden = torch.mm(hidden_list[0], ortho_weight[0])
        for i in range(1, self.num_views):  # 这操作，是否从0开始迭代就行了，可能是为了单视图版本的兼容性？[]和[[]]
            ortho_weight.append(ortho_norm(self.weight[i]))
            hidden_list[i] = hidden_list[i] - hidden_list[i].mean(dim=0)
            hidden += torch.mm(hidden_list[i], ortho_weight[i])
        # TODO 还需要激活吗，为什么没有平均1/V
        output = torch.spmm(flt_f, hidden)
        return self.activation(output), ortho_weight


class MvGCN(nn.Module):
    def __init__(self, hidden_dims, num_views, dropout):
        """单个视图里面的GCN, 这是核心组件

        Args:
            hidden_dims (_type_): _description_
            num_views (_type_): _description_
            dropout (_type_): _description_
        """        
        super(MvGCN, self).__init__()
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.num_views = num_views
        # 两层GCN
        self.gc1 = GraphConvSparse(self.hidden_dims[0], self.hidden_dims[1], self.num_views)
        self.gc2 = GraphConvSparse(self.hidden_dims[1], self.hidden_dims[2], self.num_views)

    def forward(self, input, flt):
        hidden = self.gc1(input, flt)
        output = self.gc2(hidden, flt)
        output = F.dropout(output, self.dropout, training=self.training)    # dropout在training为val和test的时候自动关闭
        return output


class IMvGCN(nn.Module):
    def __init__(self, input_dims, num_classes, dropout, layers, device):
        """
        Args:
            input_dims (list): [view0_input_dim, view1_input_dim, ...]
            num_classes (int): 类别数
            dropout (float): _description_
            layers (list): 两层的图卷积，两个隐层的维度[args.dim1, args.dim2]
            device (_type_): _description_
        """        
        super(IMvGCN, self).__init__()
        self.device = device
        self.input_dims = input_dims
        self.num_views = len(input_dims)
        self.mv_module = nn.ModuleList()    # 级联的网络架构
        hidden_dim = []
        for i in range(self.num_views):
            temp_dims = []
            temp_dims.append(input_dims[i])
            # 如dim1==8， dim2==32， imput_dim==512
            # TODO: 搞清楚这里的dim的含义            
            temp_dims.append(input_dims[i] // layers[0] if (input_dims[i] // layers[0]) >= num_classes else num_classes)
            temp_dims.append(input_dims[i] // layers[1] if (input_dims[i] // layers[1]) >= num_classes else num_classes)
            hidden_dim.append(temp_dims[-1])
            print(temp_dims)
            # 每个view的两层GCN网络，用的是Fv
            self.mv_module.append(MvGCN(hidden_dims=temp_dims, num_views=self.num_views, dropout=dropout))
        # 输出Z之后的最后卷积，用Ff，也不用激活
        self.fusion_module = FGCN(hidden_dim, num_classes, self.num_views)

    def forward(self, feature_list, flt_list, flt_f):
        hidden_list = []
        w_list = []
        for i in range(self.num_views):
            # 第i个视图的GCN使用第i视图数据和i视图的核
            hidden = self.mv_module[i](feature_list[i], flt_list[i])
            hidden_list.append(hidden)  # 每个视图的两层卷积输出
            # 每个视图的两层的正交权重
            w_list.append(self.mv_module[i].gc1.ortho_weight)
            w_list.append(self.mv_module[i].gc2.ortho_weight)
        # 名调用默认调用forward，得到的公共的Z
        common_feature, ortho_weight = self.fusion_module(hidden_list, flt_f)
        w_list += ortho_weight

        return common_feature, hidden_list, w_list