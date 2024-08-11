import time
import argparse
import pickle
import os
import logging
import numpy as np
from sessionG import *
from utils import *
from baselines import *
import os
import logging
import pandas as pd
import ast
import torch

current_dir = os.getcwd()
print("Running on ", current_dir)
path = os.path.join(current_dir, "data", "nft")
print(path)

import time

logging.basicConfig(
    filename=current_dir + f"/log/{str(int(time.time()))}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info("*******start1***********")
logging.info("start time: %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--datapath", default=path, help="path to the dataset")
parser.add_argument(
    "--dataset", default="popular_20", help="popular_20"
)  # popular_20 NFT dataset
parser.add_argument(
    "--model", default="IHRL", help="[IHRL,SASRec,GRU4Rec,CORE,SINE,LightGCN]"
)
parser.add_argument(
    "--hiddenSize", type=int, default=320
)  # 100, 96, 768  text_vec   96，6*64=384, 640 dim
parser.add_argument(
    "--n_factor", type=int, default=5, help="Disentangle factors number"
)
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--activate", type=str, default="relu")
parser.add_argument("--n_sample_all", type=int, default=4)
parser.add_argument("--n_sample", type=int, default=12)
parser.add_argument(
    "--nonhybrid", action="store_true", help="only use the global preference to predict"
)
parser.add_argument("--w", type=int, default=6, help="max window size")
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument(
    "--lr", type=float, default=0.001, help="learning rate."
)  # 0.001, 0.005,0.00025
parser.add_argument("--lr_dc", type=float, default=0.1, help="learning rate decay.")
parser.add_argument(
    "--lr_dc_step",
    type=int,
    default=3,
    help="the number of steps after which the learning rate decay.",
)
parser.add_argument("--l2", type=float, default=1e-5, help="l2 penalty ")
parser.add_argument("--layer", type=int, default=1, help="the number of layer used")
parser.add_argument("--n_iter", type=int, default=1)
parser.add_argument("--seed", type=int, default=2024)  # [1, 2]
parser.add_argument(
    "--dropout_gcn", type=float, default=0, help="Dropout rate."
)  # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument(
    "--dropout_local", type=float, default=0, help="Dropout rate."
)  # [0, 0.5]
parser.add_argument("--dropout_global", type=float, default=0.1, help="Dropout rate.")
parser.add_argument("--e", type=float, default=0.5, help="Disen H sparsity.")
parser.add_argument(
    "--disen", action="store", default=True, help="use disentangle"
)  # intent
parser.add_argument("--lamda", type=float, default=1e-4, help="aux loss weight")
parser.add_argument("--norm", action="store_true", help="use norm")
parser.add_argument(
    "--sw_edge", action="store_true", default=False, help="slide_window_edge"
)
parser.add_argument("--item_edge", action="store", default=True, help="item_edge")
parser.add_argument("--validation", action="store_true", help="validation")
parser.add_argument(
    "--valid_portion", type=float, default=0.1, help="split the portion"
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)
parser.add_argument("--patience", type=int, default=3)
parser.add_argument(
    "--price_seq", action="store", default=True, help="use trans_counts"
)
parser.add_argument(
    "--trans_counts", action="store", default=True, help="use trans_counts"
)
parser.add_argument("--nhead", type=float, default=2, help="multi_head number")
parser.add_argument(
    "--input_dim",
    type=float,
    default=50,
    help="input_dim for MA equeal to user_price_length",
)
parser.add_argument(
    "--hidden_dim",
    type=int,
    default=128,
    help="The inner hidden size in transform layer",
)  # 100, 96, 768  text_vec   96，6*64=384, 640
parser.add_argument(
    "--dropout_tran", type=float, default=0.3, help="dropout rate"
)  # 100, 96, 768  text_vec   96，6*64=384, 640
parser.add_argument(
    "--trans",
    action="store",
    default=False,
    help="if use transformer for user embedding",
)
parser.add_argument(
    "--max_seq_length", type=float, default=50, help="max_seq_length for sequeen"
)
# for sasrec
parser.add_argument(
    "--inner_size",
    type=int,
    default=256,
    help="The inner hidden size in feed-forward layer",
)
parser.add_argument(
    "--hidden_dropout_prob",
    type=float,
    default=0.5,
    help="The probability of an element to be zeroed",
)
parser.add_argument(
    "--attn_dropout_prob",
    type=float,
    default=0.5,
    help="The probability of an attention score to be zeroed",
)
parser.add_argument("--hidden_act", type=str, default="gelu")
parser.add_argument(
    "--layer_norm_eps",
    type=float,
    default=1e-12,
    help="A value added to the denominator for numerical stability.",
)
parser.add_argument(
    "--initializer_range",
    type=float,
    default=0.02,
    help="The standard deviation for normal initialization.",
)
parser.add_argument(
    "--n_layers",
    type=int,
    default=2,
    help="The number of transformer layers in transformer encoder.",
)
parser.add_argument(
    "--n_heads",
    type=int,
    default=2,
    help="The number of attention heads for multi-head attention layer.",
)
# for gru4rec
parser.add_argument(
    "--num_layers", type=int, default=1, help="The number of layers in GRU4Rec."
)
parser.add_argument(
    "--dropout_prob", type=float, default=0.3, help="The dropout rate for GRU4Rec."
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=128,
    help="The inner hidden size in feed-forward layer",
)
# for core
parser.add_argument("--dnn_type", type=str, default="trm")
parser.add_argument(
    "--item_dropout",
    type=float,
    default=0.2,
    help="The probability of candidate item embeddings to be zeroed.",
)
parser.add_argument(
    "--sess_dropout",
    type=float,
    default=0.2,
    help="The probability of item embeddings in a session to be zeroed.",
)
parser.add_argument(
    "--temperature", type=float, default=0.07, help="Temperature for contrastive loss."
)
# for sine
parser.add_argument(
    "--prototype_size",
    type=int,
    default=50,
    help="The value added to the denominator for numerical stability.",
)
parser.add_argument(
    "--interest_size", type=int, default=3, help="The number of intentions"
)
parser.add_argument(
    "--tau_ratio", type=float, default=0.1, help="The tau value for temperature tuning"
)
parser.add_argument(
    "--reg_loss_ratio", type=float, default=0.5, help="The L2 regularization weight."
)
# for lightgcn
parser.add_argument(
    "--n_users",
    type=int,
    default=29004,
    help="The value added to the denominator for numerical stability.",
)
parser.add_argument(
    "--require_pow",
    action="store_true",
    default=False,
    help="The value added to the denominator for numerical stability.",
)

opt = parser.parse_args()
print("disen", opt.disen)
save_PATH = os.path.join(current_dir, "model", "nft_model.pt")

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
# device=torch.device("cuda:{}".format(0))
CUDA_LAUNCH_BLOCKING = 1


def parse_list(data):
    try:
        return ast.literal_eval(data)
    except:
        return []


def main():
    exp_seed = opt.seed
    top_K = [5, 10, 15, 20]
    init_seed(exp_seed)

    sw = []
    for i in range(2, opt.w + 1):
        sw.append(i)

    if opt.dataset == "popular_20":
        num_node = 2254
        opt.n_iter = 1
        opt.dropout_gcn = 0.4
        opt.dropout_local = 0.0
        n_users = 29004

    logging.info(">>SEED:{}".format(exp_seed))
    logging.info("===========config================")
    logging.info("model:{}".format(opt.model))
    logging.info("dataset:{}".format(opt.dataset))
    logging.info("gpu:{}".format(opt.gpu_id))
    logging.info("Disentangle:{}".format(opt.disen))
    logging.info("Intent factors:{}".format(opt.n_factor))
    logging.info("item_edge:{}".format(opt.item_edge))
    logging.info("sw_edge:{}".format(opt.sw_edge))
    logging.info("Test Topks{}:".format(top_K))
    logging.info(f"Slide Window:{sw}")
    logging.info("hiddenSize:{}".format(opt.hiddenSize))
    logging.info("n_factor:{}".format(opt.n_factor))
    logging.info("lr:{}".format(opt.lr))
    logging.info("batch_size:{}".format(opt.batch_size))
    logging.info("===========end===================")

    datapath = path

    train_file = pickle.load(
        open(datapath + "/{}/new/train.txt".format(opt.dataset), "rb")
    )
    item_feature_file = pd.read_csv(
        datapath + "/{}/new/item_feature.csv".format(opt.dataset),
        sep=",",
        converters={"price": ast.literal_eval},
    )

    nft_prices, nft_counts = item_feature_pro(item_feature_file, max_len=30)
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_file = pickle.load(
            open(datapath + "/{}/new/test.txt".format(opt.dataset), "rb")
        )

    train_data = Data(train_file, opt, n_node=num_node, sw=sw)
    test_data = Data(test_file, opt, n_node=num_node, sw=sw)

    if opt.model == "IHRL":
        model = trans_to_cuda(IHRL(opt, num_node))
    elif opt.model == "SASRec":
        model = trans_to_cuda(SASRec(opt, num_node))
    elif opt.model == "GRU4Rec":
        model = trans_to_cuda(GRU4Rec(opt, num_node))
    elif opt.model == "CORE":
        model = trans_to_cuda(CORE(opt, num_node))
    elif opt.model == "SINE":
        model = trans_to_cuda(SINE(opt, num_node))
    elif opt.model == "LightGCN":
        inter_file = pd.read_csv(
            datapath + "/{}/new/nft.inter".format(opt.dataset), sep="\t"
        )  # 完整序列的滑动序列
        inter_mat = inter_matrix(n_users, inter_file)
        model = trans_to_cuda(LightGCN(opt, num_node, inter_mat))

    start = time.time()

    best_results = {}
    for K in top_K:
        best_results["epoch%d" % K] = [0, 0, 0]
        best_results["metric%d" % K] = [0, 0, 0]

    bad_counter = 0

    for epoch in range(opt.epoch):
        print("-------------------------------------------------------")
        logging.info(f"EPOCH:{epoch}")
        logging.info(f'Time:{time.strftime("%Y/%m/%d %H:%M:%S")}')
        total_loss, rec_loss, metrics = train_test(
            model, train_data, test_data, nft_prices, nft_counts, top_K, opt
        )
        torch.save(
            {
                "opt": opt,
                "num_node": num_node,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
                "loss": rec_loss,
            },
            save_PATH,
        )
        logging.info("total_loss: {}, rec_loss: {} ".format(total_loss, rec_loss))
        # flag = 0
        for K in top_K:
            # metrics['precision%d' % K] = np.mean(metrics['precision%d' % K]) * 100
            metrics["hit%d" % K] = np.mean(metrics["hit%d" % K]) * 100
            metrics["mrr%d" % K] = np.mean(metrics["mrr%d" % K]) * 100
            metrics["itemcoverage%d" % K] = np.mean(metrics["itemcoverage%d" % K]) * 100

            if best_results["metric%d" % K][0] < metrics["hit%d" % K]:
                best_results["metric%d" % K][0] = metrics["hit%d" % K]
                best_results["epoch%d" % K][0] = epoch
                flag = 1
            if best_results["metric%d" % K][1] < metrics["mrr%d" % K]:
                best_results["metric%d" % K][1] = metrics["mrr%d" % K]
                best_results["epoch%d" % K][1] = epoch
                flag = 1
            if best_results["metric%d" % K][2] < metrics["itemcoverage%d" % K]:
                best_results["metric%d" % K][2] = metrics["itemcoverage%d" % K]
                best_results["epoch%d" % K][2] = epoch
                flag = 1

        for K in top_K:
            logging.info("Current Result:")
            logging.info(
                "\tPrecision%d: %.4f\tMRR%d: %.4f\tIC%d: %.4f"
                % (
                    K,
                    metrics["hit%d" % K],
                    K,
                    metrics["mrr%d" % K],
                    K,
                    metrics["itemcoverage%d" % K],
                )
            )
            logging.info("Best Result:")
            logging.info(
                "\tPrecision%d: %.4f\tMRR%d: %.4f\tIC%d: %.4f\tEpoch: %d, %d, %d"
                % (
                    K,
                    best_results["metric%d" % K][0],
                    K,
                    best_results["metric%d" % K][1],
                    K,
                    best_results["metric%d" % K][2],
                    best_results["epoch%d" % K][0],
                    best_results["epoch%d" % K][1],
                    best_results["epoch%d" % K][2],
                )
            )
            bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    logging.info("-------------------------------------------------------")
    end = time.time()
    logging.info("Run time: %f s" % (end - start))


if __name__ == "__main__":
    main()
