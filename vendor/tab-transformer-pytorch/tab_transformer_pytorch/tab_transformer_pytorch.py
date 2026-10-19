import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat

from x_mlps_pytorch import MLP

from hyper_connections import mHC

from discrete_continuous_embed_readout import Embed

# helpers
# ====================== 辅助工具函数 ======================
def exists(val):
    """判断变量是否不为None"""
    return val is not None

def default(val, d):
    """如果val不为None返回val，否则返回默认值d"""
    return val if exists(val) else d

# classes
# ====================== 基础模块定义 ======================
class PreNorm(Module):
    """前置归一化层：先对输入做LayerNorm，再送入后续网络分支（Attention/FFN）"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn  # 待包裹的网络函数/模块（注意力或者前馈网络）

    def forward(self, x, **kwargs):
        # 先归一化，再执行分支网络，透传其余参数
        return self.fn(self.norm(x), **kwargs)

# attention
# ====================== Transformer内部组件 ======================
class GEGLU(Module):
    """GEGLU激活：将输入切分成两路，一路作为门控乘Gelu激活后的另一路"""
    def forward(self, x):
        # 在最后一维对半切分
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(Module):
    """Transformer前馈网络FFN，使用GEGLU结构"""
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), # 升维，为GEGLU准备两路输出
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim) # 降维回原始维度
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(Module):
    """多头自注意力模块，返回输出 + softmax后的注意力矩阵（用于可视化/分析）"""
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads # 所有头拼接后的总维度
        self.heads = heads
        self.scale = dim_head ** -0.5 # Attention缩放因子 1/sqrt(d_k)

        # 一次性映射得到 q,k,v
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim) # 多头输出融合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        # 得到q k v
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # 拆分多头: b n (h d) -> b h n d
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 计算注意力分数
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        # 加权求和value
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        # 合并多头: b h n d -> b n (h d)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out), attn

# transformer
# ====================== Transformer主体（带multi hyper connection残差流） ======================
class Transformer(Module):
    """
    多层Transformer Encoder
    使用mHC多残差流（hyper connections）替换普通残差连接
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
        self.layers = ModuleList([])
        # 获取mHC的初始化函数、流扩展函数、流压缩合并函
        init_hyper_conn, self.expand_streams, self.reduce_streams = mHC.get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        for _ in range(depth):
            self.layers.append(ModuleList([
                # Attention分支，包装PreNorm + Attention，接入mHC残差
                init_hyper_conn(dim = dim, branch = PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                # FFN分支，包装PreNorm + FeedForward，接入mHC残差
                init_hyper_conn(dim = dim, branch = PreNorm(dim, FeedForward(dim, dropout = ff_dropout))),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []  # 保存每一层注意力矩阵
        x = self.expand_streams(x)  # 将输入扩展为多残差流

        for attn, ff in self.layers:
            x, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = ff(x)

        x = self.reduce_streams(x) # 合并多残差流，变回单路张量


        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# main class
# ====================== TabTransformer主模型 ======================
class TabTransformer(Module):
    """
    TabTransformer 表格Transformer模型
    处理表格数据：类别特征送入Transformer做列自注意力，连续特征单独归一化，最后拼接送入MLP预测
    支持共享类别嵌入、特殊token、mHC多残差流
    """
    def __init__(
        self,
        *,
        categories, # list，每个类别特征的取值个数，如[5,10]代表2个类别列，分别有5、10种取值
        num_continuous, # 连续特征的列数量
        dim, # Transformer隐层维度
        depth, # Transformer层数
        heads, # 注意力头数
        dim_head = 16, # 每个注意力头的维度
        dim_out = 1, # 最终输出维度，回归设1，分类设类别数
        mlp_hidden_mults = (4, 2), # 输出MLP各层相对于输入的倍数
        mlp_act = None, # MLP激活函数
        num_special_tokens = 2, # 特殊token数量（用于缺失值等）
        continuous_mean_std = None, # 连续特征预计算的均值标准差，shape [num_continuous,2]
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_shared_categ_embed = True, # 是否开启类别共享嵌入
        shared_categ_dim_divisor = 8., # 论文中：共享嵌入维度 = dim / 8
        num_residual_streams = 4 # mHC残差流数量
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations
        # 类别特征相关统计
        self.num_categories = len(categories) # 类别特征列的数量
        self.num_unique_categories = sum(categories) # 所有类别特征总取值数量

        # take care of shared category embed
        # 共享类别嵌入
        shared_embed_dim = 0 if not use_shared_categ_embed else int(dim // shared_categ_dim_divisor)

        self.use_shared_categ_embed = use_shared_categ_embed

        if use_shared_categ_embed:
            # 每一列类别特征单独一个共享向量
            self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std = 0.02)

        # discrete embeds with groups
        # 离散类别嵌入层（分组Embed）
        self.categorical_embeds = Embed(dim - shared_embed_dim, num_discrete = categories)

        # continuous
        # 连续特征部分
        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            if exists(continuous_mean_std):
                assert continuous_mean_std.shape == (num_continuous, 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
            self.register_buffer('continuous_mean_std', continuous_mean_std)
            self.norm = nn.LayerNorm(num_continuous)

        # create category embeddings table
        # 特殊token嵌入，用于缺失值等特殊标记
        self.num_special_tokens = num_special_tokens
        self.special_token_embed = nn.Embedding(num_special_tokens, dim - shared_embed_dim)

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

        # mlp to logits
        # 最终MLP输入维度：展平后的类别Transformer输出 + 连续特征
        input_size = (dim * self.num_categories) + num_continuous
        hidden_dimensions = [input_size * t for t in  mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        self.mlp = MLP(*all_dimensions, activation = mlp_act)

    def forward(self, x_categ, x_cont, return_attn = False):
        """
        前向传播
        x_categ: 类别特征张量 shape [batch, num_categories]，负数代表特殊token
        x_cont: 连续特征张量 shape [batch, num_continuous]
        return_attn: 是否返回每一层注意力矩阵
        """
        xs = [] # 保存所有分支输出，最后拼接

        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        if self.num_unique_categories > 0:
            # 判断哪些位置是特殊token（输入<0代表缺失等特殊标记）
            is_special_token = x_categ < 0
            # clamp_min保证索引>=0，送入普通类别嵌入
            categ_embed = self.categorical_embeds(x_categ.clamp_min(0), sum_discrete_sets = False)

            if is_special_token.any():
                # 将负的特殊token id映射到0~num_special_tokens-1
                special_token_ids = (x_categ + 1).abs().clamp_max(self.num_special_tokens - 1) # use -1, -2, ... for specials
                special_embed = self.special_token_embed(special_token_ids)
                # 特殊token位置替换为special_embed，普通类别保留原有嵌入
                categ_embed = torch.where(is_special_token[..., None], special_embed, categ_embed)

            # 拼接每一列类别特征的共享嵌入
            if self.use_shared_categ_embed:
                shared_categ_embed = repeat(self.shared_category_embed, 'n d -> b n d', b = categ_embed.shape[0])
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim = -1)

            # Transformer列自注意力
            x, attns = self.transformer(categ_embed, return_attn = True)
            # 将所有类别列的embedding展平
            flat_categ = rearrange(x, 'b ... -> b (...)')
            xs.append(flat_categ)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if self.num_continuous > 0:
            # 如果提供均值标准差，则做标准化
            if exists(self.continuous_mean_std):
                mean, std = self.continuous_mean_std.unbind(dim = -1)
                x_cont = (x_cont - mean) / std

            normed_cont = self.norm(x_cont)
            xs.append(normed_cont)

        # 拼接类别分支与连续特征
        x = torch.cat(xs, dim = -1)
        logits = self.mlp(x)

        if not return_attn:
            return logits

        return logits, attns
