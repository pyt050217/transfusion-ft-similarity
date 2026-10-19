import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat

from hyper_connections import mHC

from discrete_continuous_embed_readout import Embed

# feedforward and attention
# ====================== 前馈网络与多头注意力模块 ======================
class GEGLU(Module):
    """GEGLU门控激活函数：将输入对半切分，一路作为门控，与Gelu激活后的另一路相乘"""
    def forward(self, x):
        # 在最后一维将张量切分为两份
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    """Transformer前馈网络FFN，内置LayerNorm与GEGLU，作为函数直接返回Sequential模块"""
    return nn.Sequential(
        nn.LayerNorm(dim), # 前置层归一化
        nn.Linear(dim, dim * mult * 2), # 升维，为GEGLU准备两路分支
        GEGLU(), # 门控激活
        nn.Dropout(dropout), # Dropout防止过拟合
        nn.Linear(dim * mult, dim) # 降维回原始隐空间维度
    )

class Attention(Module):
    """多头自注意力模块，内部自带LayerNorm，返回输出张量与softmax后的注意力矩阵"""
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads  # 所有注意力头拼接后的总维度
        self.heads = heads
        self.scale = dim_head ** -0.5 # 注意力缩放因子：1/sqrt(d_k)，防止点积过大

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)# 一次性映射q/k/v
        self.to_out = nn.Linear(inner_dim, dim, bias = False)# 多头输出融合映射
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        x = self.norm(x) # 先做归一化（Pre-Norm架构）
        # 拆分多头：b n (h d) -> b h n d，b批次，h头，n序列长度，d单头维度
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale
        # 计算q和k的点积相似度
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim = -1)  # 在key维度做softmax得到注意力权重
        dropped_attn = self.dropout(attn)

        # 使用注意力权重对v加权求和
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        # 合并多头：b h n d -> b n (h d)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer
# ====================== Transformer编码器主体（mHC多残差流） ======================
class Transformer(Module):
    """
    多层Transformer Encoder
    使用mHC（multi Hyper Connections）多残差流替代普通残差连接，提升深层网络训练稳定性
    """
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        num_residual_streams = 4
    ):
        super().__init__()
        # 获取mHC的分支初始化函数、流扩展函数、流压缩合并函数
        init_hyper_conn, self.expand_streams, self.reduce_streams = mHC.get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                # 注意力分支，接入mHC残差连接
                init_hyper_conn(dim = dim, branch = Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                # 前馈分支，接入mHC残差连接
                init_hyper_conn(dim = dim, branch = FeedForward(dim, dropout = ff_dropout)),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []  # 保存每一层的注意力矩阵，用于可视化分析
        x = self.expand_streams(x)  # 将输入扩展为多路残差流

        for attn, ff in self.layers:
            x, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = ff(x)

        x = self.reduce_streams(x) # 合并多路残差流，恢复成单路张量

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# numerical embedder
# ====================== 数值特征嵌入模块 ======================
class NumericalEmbedder(Module):
    """FT-Transformer原版数值嵌入：为每一列连续特征学习一组权重和偏置，将标量转为向量token"""
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1') # 增加最后一维，方便广播相乘
        return x * self.weights + self.biases

# main class
# ====================== FTTransformer主模型 ======================
class FTTransformer(Module):
    """
    FT-Transformer 表格Transformer（Feature Tokenizer Transformer）
    论文核心：**每一列特征（类别/连续）都变成一个独立token**，加上CLS token送入Transformer；
    最终用CLS token做预测，是表格深度学习SOTA模型。
    """
    def __init__(
        self,
        *,
        categories, # list，每个类别特征的取值数量，如[3,5]代表2个类别列
        num_continuous, # 连续特征的列数量
        dim, # 特征token隐向量维度
        depth, # Transformer层数
        heads, # 注意力头数量
        dim_head = 16, # 单个注意力头维度
        dim_out = 1, # 输出维度，回归任务=1；分类=类别总数
        num_special_tokens = 2, # 特殊token数量，用于缺失值等
        attn_dropout = 0., # 注意力层dropout
        ff_dropout = 0., # FFN层dropout
        num_residual_streams = 4 # mHC多残差流数量
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations
        # 类别特征相关统计
        self.num_categories = len(categories) # 类别特征的列数
        self.num_unique_categories = sum(categories) # 所有类别特征原始取值总数

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # embed
        # 给每个类别列预留特殊token位置（用于缺失值）
        categories_with_special = tuple(c + num_special_tokens for c in categories)
        # 嵌入模块：同时处理离散类别特征 + 连续数值特征
        self.embedding = Embed(dim, num_discrete = categories_with_special, num_continuous = num_continuous, auto_append_discrete_set_dim = False)

        # cls token
        # CLS token：全局聚合token，放在序列最前面，最后用于预测
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer编码器
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            num_residual_streams = num_residual_streams
        )

        # to logits
        # 输出预测头：LN -> ReLU -> Linear，论文原版结构
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn = False):
        """
        前向传播函数
        x_categ: 类别特征，shape [batch, num_categories]
        x_numer: 连续数值特征，shape [batch, num_continuous]
        return_attn: 是否返回各层注意力矩阵（用于可视化/分析）
        """
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        # 类别id偏移：留出前面num_special_tokens位置给特殊token（缺失值）
        x_categ = x_categ + self.num_special_tokens

        # 类别、数值特征分别嵌入，拼接成token序列
        x = self.embedding(
            (x_categ, x_numer),
            sum_discrete_sets = False,
            sum_continuous = False,
            concat_discrete_continuous = True
        )

        # append cls tokens
        # 在序列开头拼接CLS token
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        # attend
        # Transformer全局自注意力
        x, attns = self.transformer(x, return_attn = True)

        # get cls token
        # 取出CLS token（序列第0位）作为全局特征用于预测
        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))
        # 预测头输出logits
        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns
