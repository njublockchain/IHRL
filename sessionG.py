import math
from metrics import calc_IC
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from layers import DisentangleGraph, LocalHyperGATlayer
from torch.nn import Module, MultiheadAttention
import torch.nn.functional as F
import torch.sparse


class IHRL(Module):
    def __init__(self, opt, num_node, adj_all=None, num=None, cat=False):
        super(IHRL, self).__init__()
        # HYPER PARA
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.n_factor = opt.n_factor
        self.sample_num = opt.n_sample
        self.nonhybrid = opt.nonhybrid
        self.layer = int(opt.layer)
        self.n_factor = opt.n_factor  # number of intention prototypes
        self.cat = cat
        self.e = opt.e
        self.disen = opt.disen
        self.nhead = self.opt.nhead
        self.w_k = 10
        self.dim = opt.hiddenSize

        # Item representation
        self.embedding = nn.Embedding(num_node, self.dim)

        if self.disen:
            self.feat_latent_dim = self.dim // self.n_factor
            self.split_sections = [self.feat_latent_dim] * self.n_factor

        else:
            self.feat_latent_dim = self.dim

        # Position representation
        self.pos_embedding = nn.Embedding(200, self.dim)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(3 * self.dim, 1))
        self.w_s = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.glu1 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu3 = nn.Linear(self.dim, self.dim, bias=True)

        self.MA_dim_user = 1  # self.opt.input_dim
        self.MA_dim_item = 30  # self.opt.input_dim
        # self.MA_price_u = MultiheadAttention(self.MA_dim_user, self.nhead)
        # self.MA_count_u = MultiheadAttention(self.MA_dim_user, self.nhead)
        self.MA_price_i = MultiheadAttention(self.MA_dim_item, self.nhead)
        self.MA_count_i = MultiheadAttention(self.MA_dim_item, self.nhead)
        if opt.price_seq:
            self.f_p_u = nn.Linear(self.MA_dim_user, self.dim)
            self.f_p_i = nn.Linear(self.MA_dim_item, self.dim)

            if opt.trans_counts:
                self.f_p_u = nn.Linear(self.MA_dim_user + 1, self.dim)
                self.f_p_i = nn.Linear(self.MA_dim_user + 1, self.dim)

        self.feature_fus_u = nn.Linear(self.dim * 2, self.dim)

        encoder_transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.nhead,
            dim_feedforward=self.opt.hidden_dim,
            dropout=self.opt.dropout_tran,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_transformer_layer, num_layers=1
        )
        ###### text representation #######

        if self.disen:
            self.disenG = DisentangleGraph(
                dim=self.feat_latent_dim, alpha=self.opt.alpha, e=self.e
            )  # need to be updated
            self.disen_aggs = nn.ModuleList(
                [
                    LocalHyperGATlayer(
                        self.feat_latent_dim,
                        self.layer,
                        self.opt.alpha,
                        self.opt.dropout_gcn,
                    )
                    for i in range(self.n_factor)
                ]
            )

            self.classifier = nn.Linear(self.feat_latent_dim, self.n_factor)
            self.loss_aux = nn.CrossEntropyLoss()
            self.intent_loss = 0
        else:
            self.local_agg = LocalHyperGATlayer(
                self.dim, self.layer, self.opt.alpha, self.opt.dropout_gcn
            )

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        # main task loss
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=opt.lr, weight_decay=opt.l2
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_disentangle_loss(self, intents_feat):
        # compute discrimination loss

        labels = [
            torch.ones(f.shape[0]) * i for i, f in enumerate(intents_feat)
        ]  # (intent_num, (batch_size, latent_dim))
        labels = trans_to_cuda(
            torch.cat(tuple(labels), 0)
        ).long()  # (batch_size*intent_num)
        intents_feat = torch.cat(
            tuple(intents_feat), 0
        )  # (batch_size*intent_num, latent_dim)

        pred = self.classifier(intents_feat)  # (batch_size*intent_num, intent_num)
        discrimination_loss = self.loss_aux(pred, labels)
        return discrimination_loss

    def compute_scores(
        self, hidden_, hidden, mask, item_embeddings
    ):  # hidden(256,49,1368), maskhidden(256,49,1), item_embeddings(14450,768)

        # select = self.transformer(hidden)
        # select = select.mean(1)

        if self.opt.trans:
            mask = mask.float().unsqueeze(-1)

            batch_size = hidden.shape[0]
            len = hidden.shape[1]
            pos_emb = self.pos_embedding.weight[:len]
            pos_emb = pos_emb.unsqueeze(0).repeat(
                batch_size, 1, 1
            )  # (b, N, dim) （100，200，100）

            hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
            hs = hs.unsqueeze(-2).repeat(1, len, 1)
            ht = hidden[:, 0, :]
            ht = ht.unsqueeze(-2).repeat(1, len, 1)  # (b, N, dim)

            nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
            nh = torch.tanh(nh)

            hs = torch.cat([hs, ht], -1).matmul(self.w_s)

            feat = hs * hidden
            nh = torch.sigmoid(
                torch.cat([self.glu1(nh), self.glu2(hs), self.glu3(feat)], -1)
            )

            beta = torch.matmul(nh, self.w_2)
            beta = beta * mask  # (256,49,1)
            select = torch.sum(beta * hidden, 1)  # (256,1368)
        else:
            select = torch.mean(hidden, 1)

        # if self.opt.price_seq:
        #     select = self.feature_fus_u(torch.cat((select, user_feature), dim=-1))
        if self.disen:
            score_all = []
            # split_sections = self.feat_latent_dim * self.n_factor   #228
            select_split = torch.split(select, self.split_sections, dim=-1)
            b = torch.split(
                item_embeddings[1:], self.split_sections, dim=-1
            )  # 6*([14449,128])
            for i in range(self.n_factor):
                sess_emb_int = self.w_k * select_split[i]  # (256,228)
                item_embeddings_int = b[i]

                scores_int = torch.mm(
                    sess_emb_int, torch.transpose(item_embeddings_int, 1, 0)
                )
                score_all.append(scores_int)

            score = torch.stack(score_all, dim=1)
            scores = score.sum(1)

        else:
            b = item_embeddings[1:]  # n_nodes x latent_size
            scores = torch.matmul(select, b.transpose(1, 0))

        return scores

    def forward(
        self,
        user_id,
        inputs,
        Hs,
        mask_item,
        item,
        user_price_seq,
        item_price_seq,
        user_count,
        nft_count,
        len_data,
    ):  # items, Hs, mask, inputs
        #  inputs:(batch,max_len)

        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        user_price_seq = user_price_seq.unsqueeze(-1)
        user_count = user_count.unsqueeze(-1)

        if self.opt.price_seq:
            #     user_price_se = user_price_seq.transpose(0, 1)
            #     user_price_se,_ = self.MA_price_u(user_price_se,user_price_se,user_price_se)

            #     if self.opt.trans_counts:
            #         user_cou = user_count.transpose(0, 1)
            #         user_count_feature,_ = self.MA_count_u(user_cou,user_cou,user_cou)

            #         user_feature =  self.f_p_u(torch.cat((user_price_se, user_count_feature), dim=-1))  #(512,64)
            #         user_feature = user_feature.transpose(0, 1)

            #     else:
            #         user_feature = user_price_se.transpose(0, 1)
            #         user_feature = self.f_p_u(user_feature)

            # item_price_seq = item_price_seq.transpose(0, 1)
            item_count_feature, _ = self.MA_price_i(
                item_price_seq, item_price_seq, item_price_seq
            )
            # item_count_feature= self.transformer(item_price_seq)
            item_count_feature = (item_count_feature.mean(1)).unsqueeze(-1)
            nft_count = nft_count.unsqueeze(-1)
            if self.opt.trans_counts:
                item_feature = self.f_p_i(
                    torch.cat((item_count_feature, nft_count), dim=-1)
                )
            else:
                item_feature = self.f_p_i(item_count_feature)

        item_embeddings = self.embedding.weight
        zeros = trans_to_cuda(torch.FloatTensor(1, self.dim).fill_(0))
        item_embeddings = torch.cat([zeros, item_embeddings], 0)  # (14451,768)

        h = item_embeddings[inputs]  # (256,49,768)
        if self.opt.price_seq:
            h = item_embeddings[1:][inputs] * item_feature[inputs]
        item_emb = item_embeddings[item] * mask_item.float().unsqueeze(
            -1
        )  # batch, node, dim

        session_c = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(
            -1
        )  # batch, dim
        session_c = session_c.unsqueeze(1)  # (batchsize, edge_num, dim)  [256,1,768]

        if self.disen:
            # intent prototypes from the clustering of all items
            all_items = item_embeddings[1:]  # item_num x dim   (14449,768)
            intents_cat = torch.mean(all_items, dim=0, keepdim=True)  # 1 x dim  (1,768)

            # Parallel disen-encoders
            mask_node = torch.ones_like(inputs)  # (256,49)
            zeor_vec = torch.zeros_like(inputs)  # (256,49)
            mask_node = torch.where(inputs.eq(0), zeor_vec, mask_node)  # (256,49)
            # 分成k个子空间
            h_split = torch.split(h, self.split_sections, dim=-1)
            s_split = torch.split(
                session_c, self.split_sections, dim=-1
            )  # 6*(256,1,128)
            intent_split = torch.split(
                intents_cat, self.split_sections, dim=-1
            )  # 6*(1,128)
            h_ints = []
            intents_feat = []

            for i in range(self.n_factor):

                h_int = h_split[i]
                int_emb_vec = intent_split[i]  # (1,128)
                int_emb = int_emb_vec.unsqueeze(0).repeat(batch_size, 1, 1)
                session_emb = s_split[i]

                Hs = self.disenG(
                    h_int, Hs, int_emb, mask_node
                )  #  construct intent hyperedges for each session
                h_int = self.disen_aggs[i](
                    h_int, Hs, session_emb
                )  # representation of each session  (256,49,228)

                # Activate disentangle with intent protypes
                intent_p = int_emb.expand(batch_size, seqs_len, int_emb.shape[-1])

                sim_val = h_int * intent_p
                cor_att = torch.sigmoid(sim_val)
                h_int = h_int * cor_att + h_int

                h_ints.append(h_int)
                intents_feat.append(torch.mean(h_int, dim=1))  # (b ,latent_dim)

            h_stack = torch.stack(
                h_ints, dim=2
            )  # (b ,len, k, latent_dim)  (256,49,6,228)
            dim_new = self.dim
            h_local = h_stack.reshape(batch_size, seqs_len, dim_new)

            # Aux task: intent prediction
            self.intent_loss = self.compute_disentangle_loss(
                intents_feat
            )  # Calculate the loss of auxiliary tasks

        else:

            h_local = self.local_agg(h, Hs, session_c)

        output = h_local

        return output, item_embeddings


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable.cpu()


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data, nft_prices, nft_counts):
    (
        user_id,
        alias_inputs,
        Hs,
        items,
        mask,
        targets,
        inputs,
        user_price_seq,
        user_count,
        len_data,
    ) = data
    user_id = trans_to_cuda(user_id).long()
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()  # （256，49）
    Hs = trans_to_cuda(Hs).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()  # （256，49）
    user_price_seq = trans_to_cuda(user_price_seq).float()
    item_price_seq = trans_to_cuda(nft_prices).float()
    # user_count = trans_to_cuda(user_count).float()
    user_count = trans_to_cuda(np.log1p(user_count)).float()
    # nft_count = trans_to_cuda(nft_count).float()
    nft_count = trans_to_cuda(np.log1p(nft_counts)).float()
    len_data = trans_to_cuda(len_data).long()

    hidden, item_embeddings = model(
        user_id,
        items,
        Hs,
        mask,
        inputs,
        user_price_seq,
        item_price_seq,
        user_count,
        nft_count,
        len_data,
    )
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    return targets, model.compute_scores(hidden, seq_hidden, mask, item_embeddings)


def train_test(model, train_data, test_data, nft_prices, nft_counts, top_K, opt):
    # print('start training: ', datetime.datetime.now())
    model.train()
    total_loss, totaloss = 0.0, 0.0
    rec_loss = 0.0
    train_loader = torch.utils.data.DataLoader(
        train_data,
        num_workers=4,
        batch_size=model.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data, nft_prices, nft_counts)
        targets = trans_to_cuda(targets).long()
        # loss = model.loss_function(scores, targets - 1)
        loss = model.loss_function(scores, targets)
        totaloss = loss
        if opt.disen:
            intent_loss = opt.lamda * model.intent_loss
            totaloss += intent_loss
        totaloss.backward()
        model.optimizer.step()
        total_loss += totaloss
        rec_loss += loss
    model.scheduler.step()

    metrics = {}
    for K in top_K:
        metrics["hit%d" % K] = []
        metrics["mrr%d" % K] = []
        metrics["itemcoverage%d" % K] = []

    model.eval()
    test_loader = torch.utils.data.DataLoader(
        test_data,
        num_workers=4,
        batch_size=model.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    max_K = max(top_K)

    recbole_items_result = []
    for data in test_loader:
        targets, scores = forward(model, data, nft_prices, nft_counts)
        targets = targets.numpy()

        for K in top_K:
            sub_scores = scores.topk(K)[1]

            sub_scores = trans_to_cpu(sub_scores).detach().numpy()

            for score, target, mask in zip(
                sub_scores, targets, test_data.mask
            ):  # Predicted results of each session
                metrics["hit%d" % K].append(
                    np.isin(target, score)
                )  # Whether the predicted result contains real items
                if len(np.where(score == target)[0]) == 0:
                    metrics["mrr%d" % K].append(0)
                else:
                    metrics["mrr%d" % K].append(
                        1 / (np.where(score == target)[0][0] + 1)
                    )

        max_K = max(top_K)
        ic_metrics = calc_IC(
            top_K, scores.topk(max_K)[1].cpu().detach(), len(nft_counts) - 1
        )
        for k in ic_metrics.keys():
            key = k.replace("@", "")
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(ic_metrics[k])

    return total_loss, rec_loss, metrics
