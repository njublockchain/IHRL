import numpy as np
import torch
import scipy.sparse as sp 
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler



def inter_matrix(n_users, inter_file):
    users = inter_file['user_id'].tolist()
    items = inter_file['item_id'].tolist()
    data = [1] * len(users)
    num_users = n_users
    num_items = len(set(items))
    mat = coo_matrix(
            (data, (users, items)), shape=(num_users, num_items)
        )
    return mat

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def map_data(Data):
    s_data = Data[0]
    s_target = Data[1]
    cur_data = []
    cur_target = []
    for i in range(len(s_data)):
        data = s_data[i]
        target = s_target[i]
        if len(data) > 32:
            continue
        cur_data.append(data)
        cur_target.append(target)
    return [cur_data, cur_target]

def handle_data(inputData, sw, opt):
    items, len_data = [], []
    for nowData in inputData:
        len_data.append(len(nowData))   # 记录每个session的长度
        Is = []
        for i in nowData:
            Is.append(i)
        items.append(Is)   # 记录每个session的item序列

    # len_data = [len(nowData) for nowData in inputData]
    max_len = max(len_data)  # 最大session长度
    max_len = 50  # 最大session长度

    edge_lens = []
    for item_seq in items:
        item_num = len(list(set(item_seq)))    # session中的item个数
        num_sw = 0
        if opt.sw_edge:
            for win_len in sw:
                temp_num = len(item_seq) - win_len + 1
                num_sw += temp_num
        edge_num = num_sw
        if opt.item_edge:
            edge_num += item_num
        edge_lens.append(edge_num)    

    max_edge_num = max(edge_lens)  # If neither is selected, the value is 0
    # reverse the sequence
    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]   # padding
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]

    #print(max_len, max_edge_num)

    return us_pois, us_msks, max_len, max_edge_num, items, len_data

def price_seq_data(data, max_len=30):
    
    us_pois = [list(reversed(price_seq)) + [0] * (max_len - len(price_seq)) if len(price_seq) < max_len else list(reversed(price_seq[-max_len:]))
               for price_seq in data]
    return us_pois

def normalize_sequence(price_seq, scaler, max_len):
    # rev sequence
    price_seq = list(reversed(price_seq))
    
    # 如果长度不足 max_len，填充0
    if len(price_seq) < max_len:
        price_seq = price_seq + [0] * (max_len - len(price_seq))
    else:
        price_seq = price_seq[:max_len]
    price_array = np.array(price_seq)
    non_zero_indices = price_array > 0
    non_zero_prices = price_array[non_zero_indices]
    
    if len(non_zero_prices) > 0:
        normalized_non_zero_prices = scaler.transform(non_zero_prices.reshape(-1, 1)).flatten()
        price_array[non_zero_indices] = normalized_non_zero_prices
    
    return price_array.tolist()

def item_feature_pro(item_feature, max_len=30):
    nft_count = np.asarray(item_feature['transaction_count'])
    item_price_seq = item_feature['price']
    # nft_prices = np.asarray([list(reversed(price_seq)) + [0] * (max_len - len(price_seq)) if len(price_seq) < max_len else list(reversed(price_seq[-max_len:]))
    #            for price_seq in item_price_norm])
    all_prices = np.concatenate(item_price_seq)
    non_zero_prices = all_prices[all_prices > 0]
    scaler = MinMaxScaler()
    scaler.fit(non_zero_prices.reshape(-1, 1))
    nft_prices = np.array([normalize_sequence(price_seq, scaler, max_len) for price_seq in item_price_seq])

    return torch.tensor(nft_prices),torch.tensor(nft_count)

class Data(Dataset):
    def __init__(self, data, opt, n_node, sw=[2]):
        self.n_node = n_node
        # inputs, mask, max_len, max_edge_num, seq_items = handle_data(data[0], sw, opt)  # 为滑动窗口准备数据，
        inputs, mask, max_len, max_edge_num, seq_items, len_data = handle_data(data['user_seqs'], sw, opt)  # 为滑动窗口准备数据，
        user_price_seq = price_seq_data(data['user_prices'], 30)
        user_count = price_seq_data(data['user_past_counts'],30)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data['item_id'])  # len: 79573
        self.user_ids = np.asarray(data['user_id'])  # 79675
        self.user_price_seq = np.asarray(user_price_seq)
        # self.item_price_seq = np.asarray(item_price_seq)
        # self.user_count = np.asarray(data['user_past_counts'])
        self.user_count = np.asarray(user_count)
        # self.nft_count = np.asarray(item_feature['transaction_count'])
        self.len_data = np.asarray(len_data)
        
        self.mask = np.asarray(mask)
        # self.length = len(data[0])
        self.length = len(data['user_seqs'])
        self.max_len = max_len # max_node_num
        self.max_edge_num = max_edge_num  # max_edge_num
        self.sw = sw # slice window
        self.opt = opt
        self.seq_items = seq_items

    def __getitem__(self, index):
        u_input, mask, target,len_data = self.inputs[index], self.mask[index], self.targets[index], self.len_data[index]
        user_id = self.user_ids[index]
        max_n_node = self.max_len
        max_n_edge = self.max_edge_num # max hyperedge num

        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0]    # padding 0
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]  # 对应u_input中的每个元素,获取它在node中的索引位置

        # H_s shape: (max_n_node, max_n_edge)
        rows = []
        cols = []
        vals = []
        # generate slide window hyperedge
        edge_idx = 0
        if self.opt.sw_edge:
            for win in self.sw:
                for i in range(len(u_input)-win+1):
                    if i+win <= len(u_input):
                        if u_input[i+win-1] == 0:
                            break
                        for j in range(i, i+win):
                            rows.append(np.where(node == u_input[j])[0][0])
                            cols.append(edge_idx)
                            vals.append(1.0)
                        edge_idx += 1
        

        if self.opt.item_edge:
            # generate in-item hyperedge, ignore 0
            for item in node:
                if item != 0:
                    for i in range(len(u_input)):
                        if u_input[i] == item and i > 0:
                            rows.append(np.where(node == u_input[i-1])[0][0])
                            cols.append(edge_idx)
                            vals.append(2.0)
                    rows.append(np.where(node == item)[0][0])
                    cols.append(edge_idx)
                    vals.append(1.0)
                    edge_idx += 1
        
        # intent hyperedges are dynamic generated in ci_layers.py
        u_Hs = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))   # (49,49)
        Hs = np.asarray(u_Hs.todense())    # Dense representation array of adjacency matrices
        user_price_seq = self.user_price_seq[index]
        user_count = self.user_count[index]


        return [torch.tensor(user_id), torch.tensor(alias_inputs), torch.tensor(Hs), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input), 
                torch.tensor(user_price_seq), torch.tensor(user_count),  torch.tensor(len_data)]
                # items, Hs, mask, inputs

    def __len__(self):
        return self.length
