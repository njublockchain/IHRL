###nft
## the proposed model's baseline
# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""
import numpy as np
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder
from torch.nn.init import xavier_uniform_, xavier_normal_


class SASRec(Module):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, opt, num_node):
        super(SASRec, self).__init__()
        # load parameters info
        self.batch_size = opt.batch_size
        self.n_layers = opt.n_layers
        self.n_heads = opt.n_heads
        self.hidden_size = opt.hiddenSize  # same as embedding_size
        self.inner_size = opt.inner_size  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = opt.hidden_dropout_prob
        self.attn_dropout_prob = opt.attn_dropout_prob
        self.hidden_act = opt.hidden_act
        self.layer_norm_eps = opt.layer_norm_eps
        self.initializer_range = opt.initializer_range
        self.n_items = num_node
        self.max_seq_length = opt.max_seq_length

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=opt.lr, weight_decay=opt.l2
        )  # 优化器
        self.intent_loss = 0
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc
        )  # 学习率衰减计划
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        user_id,
        items,
        Hs,
        mask,
        item_seq,
        user_price_seq,
        item_price_seq,
        user_count,
        nft_count,
        item_seq_len,
    ):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = gather_indexes(output, item_seq_len - 1)

        return output, self.item_embedding.weight

    def compute_scores(self, output, seq_hidden, mask, item_embedding):
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


def get_attention_mask(item_seq, bidirectional=False):
    """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
    attention_mask = item_seq != 0
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
    if not bidirectional:
        extended_attention_mask = torch.tril(
            extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
        )
    extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
    return extended_attention_mask


def gather_indexes(output, gather_index):
    """Gathers the vectors at the specific positions over a minibatch"""
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)


class GRU4Rec(Module):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.
    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, opt, num_node):
        super(GRU4Rec, self).__init__()

        # load parameters info
        self.batch_size = opt.batch_size
        self.embedding_size = opt.hiddenSize
        self.hidden_size = opt.hidden_size
        self.num_layers = opt.num_layers
        self.dropout_prob = opt.dropout_prob
        self.n_items = num_node
        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=opt.lr, weight_decay=opt.l2
        )  # 优化器
        self.intent_loss = 0
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc
        )  # 学习率衰减计划

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(
        self,
        user_id,
        items,
        Hs,
        mask,
        item_seq,
        user_price_seq,
        item_price_seq,
        user_count,
        nft_count,
        item_seq_len,
    ):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        output = gather_indexes(gru_output, item_seq_len - 1)

        return output, self.item_embedding.weight

    def compute_scores(self, output, seq_hidden, mask, item_embedding):
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


# -*- coding: utf-8 -*-


class TransNet(nn.Module):
    def __init__(self, opt, num_node):
        super().__init__()

        self.n_layers = opt.n_layers
        self.n_heads = opt.n_heads
        self.hidden_size = opt.hiddenSize
        self.inner_size = opt.inner_size
        self.hidden_dropout_prob = opt.hidden_dropout_prob
        self.attn_dropout_prob = opt.attn_dropout_prob
        self.hidden_act = opt.hidden_act
        self.layer_norm_eps = opt.layer_norm_eps
        self.initializer_range = opt.initializer_range

        # self.position_embedding = nn.Embedding(
        #     dataset.field2seqlen[config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]],
        #     self.hidden_size,
        # )
        self.position_embedding = nn.Embedding(num_node, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.fn = nn.Linear(self.hidden_size, 1)

        self.apply(self._init_weights)

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]

        alpha = self.fn(output).to(torch.double)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        return alpha

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class CORE(Module):
    r"""CORE is a simple and effective framewor, which unifies the representation spac
    for both the encoding and decoding processes in session-based recommendation.
    Reference:
        Yupeng Hou, Binbin Hu, Zhiqiang Zhang, Wayne Xin Zhao. "CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space." in SIGIR 2022.
        https://github.com/RUCAIBox/CORE
    """

    def __init__(self, opt, num_node):
        super(CORE, self).__init__()

        # load parameters info
        self.batch_size = opt.batch_size
        self.embedding_size = opt.hiddenSize
        self.dnn_type = opt.dnn_type
        self.sess_dropout = nn.Dropout(opt.sess_dropout)
        self.item_dropout = nn.Dropout(opt.item_dropout)
        self.temperature = opt.temperature
        self.n_items = num_node

        # item embedding
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )

        # DNN
        if self.dnn_type == "trm":
            self.net = TransNet(opt, num_node)
        elif self.dnn_type == "ave":
            self.net = self.ave_net
        else:
            raise ValueError(
                f"dnn_type should be either trm or ave, but have [{self.dnn_type}]."
            )

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=opt.lr, weight_decay=opt.l2
        )  # 优化器
        self.intent_loss = 0
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc
        )  # 学习率衰减计划
        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @staticmethod
    def ave_net(item_seq, item_emb):
        mask = item_seq.gt(0)
        alpha = mask.to(torch.float) / mask.sum(dim=-1, keepdim=True)
        return alpha.unsqueeze(-1)

    def forward(
        self,
        user_id,
        items,
        Hs,
        mask,
        item_seq,
        user_price_seq,
        item_price_seq,
        user_count,
        nft_count,
        item_seq_len,
    ):
        x = self.item_embedding(item_seq)
        x = self.sess_dropout(x)
        # Representation-Consistent Encoder (RCE)
        alpha = self.net(item_seq, x)
        seq_output = torch.sum(alpha * x, dim=1)
        output = F.normalize(seq_output, dim=-1)
        return output, self.item_embedding.weight

    def compute_scores(self, output, seq_output, mask, item_embedding):
        test_items_emb = self.item_embedding.weight
        test_item_emb = F.normalize(test_items_emb, dim=-1)
        scores = torch.matmul(output, test_item_emb.transpose(0, 1)) / self.temperature
        return scores


from recbole.utils import InputType

torch.autograd.set_detect_anomaly(True)


class SINE(Module):
    r"""################################################
    Reference:
        Qiaoyu Tan et al. "Sparse-Interest Network for Sequential Recommendation." in WSDM 2021.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, opt, num_node):
        super(SINE, self).__init__()

        # load dataset info
        # self.n_users = dataset.user_num
        self.n_items = num_node

        # load parameters info
        self.batch_size = opt.batch_size
        self.embedding_size = opt.hiddenSize
        self.layer_norm_eps = opt.layer_norm_eps

        self.D = opt.hiddenSize
        self.L = opt.prototype_size  # 500 for movie-len dataset
        self.k = opt.interest_size  # 4 for movie-len dataset
        self.tau = opt.tau_ratio  # 0.1 in paper
        self.max_seq_length = opt.max_seq_length

        self.initializer_range = 0.01

        self.w1 = self._init_weight((self.D, self.D))
        self.w2 = self._init_weight(self.D)
        self.w3 = self._init_weight((self.D, self.D))
        self.w4 = self._init_weight(self.D)

        self.C = nn.Embedding(self.L, self.D)

        self.w_k_1 = self._init_weight((self.k, self.D, self.D))
        self.w_k_2 = self._init_weight((self.k, self.D))
        self.item_embedding = nn.Embedding(self.n_items, self.D, padding_idx=0)
        self.ln2 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.ln4 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=opt.lr, weight_decay=opt.l2
        )
        self.intent_loss = 0
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc
        )

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weight(self, shape):
        mat = torch.FloatTensor(np.random.normal(0, self.initializer_range, shape))
        return nn.Parameter(mat, requires_grad=True)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        user_id,
        items,
        Hs,
        mask,
        item_seq,
        user_price_seq,
        item_price_seq,
        user_count,
        nft_count,
        item_seq_len,
    ):
        device = item_seq.device
        x_u = self.item_embedding(item_seq).to(device)  # [B, N, D]

        # concept activation
        # sort by inner product
        x = torch.matmul(x_u, self.w1)
        x = torch.tanh(x)
        x = torch.matmul(x, self.w2)
        a = F.softmax(x, dim=1)
        z_u = torch.matmul(a.unsqueeze(2).transpose(1, 2), x_u).transpose(1, 2)
        s_u = torch.matmul(self.C.weight, z_u)
        s_u = s_u.squeeze(2)
        idx = s_u.argsort(1)[:, -self.k :]
        s_u_idx = s_u.sort(1)[0][:, -self.k :]
        c_u = self.C(idx)
        sigs = torch.sigmoid(s_u_idx.unsqueeze(2).repeat(1, 1, self.embedding_size))
        C_u = c_u.mul(sigs)

        # intention assignment
        # use matrix multiplication instead of cos()
        w3_x_u_norm = F.normalize(x_u.matmul(self.w3), p=2, dim=2)
        C_u_norm = self.ln2(C_u)
        P_k_t = torch.bmm(w3_x_u_norm, C_u_norm.transpose(1, 2))
        P_k_t_b = F.softmax(P_k_t, dim=2)
        P_k_t_b_t = P_k_t_b.transpose(1, 2)

        # attention weighting
        a_k = x_u.unsqueeze(1).repeat(1, self.k, 1, 1).matmul(self.w_k_1)
        P_t_k = F.softmax(
            torch.tanh(a_k)
            .matmul(self.w_k_2.reshape(self.k, self.embedding_size, 1))
            .squeeze(3),
            dim=2,
        )

        # interest embedding generation
        mul_p = P_k_t_b_t.mul(P_t_k)
        x_u_re = x_u.unsqueeze(1).repeat(1, self.k, 1, 1)
        mul_p_re = mul_p.unsqueeze(3)
        delta_k = x_u_re.mul(mul_p_re).sum(2)
        delta_k = F.normalize(delta_k, p=2, dim=2)

        # prototype sequence
        x_u_bar = P_k_t_b.matmul(C_u)
        C_apt = F.softmax(torch.tanh(x_u_bar.matmul(self.w3)).matmul(self.w4), dim=1)
        C_apt = C_apt.reshape(-1, 1, self.max_seq_length).matmul(x_u_bar)
        C_apt = self.ln4(C_apt)

        # aggregation weight
        e_k = delta_k.bmm(C_apt.reshape(-1, self.embedding_size, 1)) / self.tau
        e_k_u = F.softmax(e_k.squeeze(2), dim=1)
        v_u = e_k_u.unsqueeze(2).mul(delta_k).sum(dim=1)

        return v_u, self.item_embedding.weight

    def compute_scores(self, output, seq_hidden, mask, item_embedding):
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores


import scipy.sparse as sp
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class LightGCN(Module):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.
    Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.
    Reference code:
    https://github.com/kuandeng/LightGCN
    We implement the model following the original author with a pairwise training mode.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, opt, num_node, inter_matrix):
        super(LightGCN, self).__init__()

        # load dataset info
        self.interaction_matrix = inter_matrix
        # load parameters info
        self.batch_size = opt.batch_size
        self.latent_dim = opt.hiddenSize  # int type:the embedding size of lightGCN
        self.n_layers = opt.n_layers  # int type:the layer num of lightGCN
        self.reg_weight = (
            opt.lamda
        )  # float32 type: the weight decay for l2 normalization
        self.require_pow = opt.require_pow
        self.n_users = opt.n_users
        self.n_items = num_node

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )

        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat()

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=opt.lr, weight_decay=opt.l2
        )  # 优化器
        self.intent_loss = 0
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc
        )  # 学习率衰减计划

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(
        self,
        user_id,
        items,
        Hs,
        mask,
        item_seq,
        user_price_seq,
        item_price_seq,
        user_count,
        nft_count,
        item_seq_len,
    ):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            norm_adj_matrix = self.norm_adj_matrix.to(all_embeddings.device)
            all_embeddings = torch.sparse.mm(norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        user_embeds = user_all_embeddings[user_id]
        return user_embeds, item_all_embeddings

    def compute_scores(self, restore_user_e, seq_hidden, mask, restore_item_e):

        u_embeddings = restore_user_e
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))

        return scores
