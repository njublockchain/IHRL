# IHRL

## Ready

We suggest using anaconda/miniconda.

Create a new python env with [pytorch](https://pytorch.org/get-started/locally/).

1. `git clone` this repo, and change diretory into the repo's root folder
2. decompress the data
```
unzip data.zip
```
3. install other dependencies from PyPI
```
pip install -r requirements.txt
```

## Usage

For re-producing our experiment result, just run `main.py` with the default parameters

```
python main.py
```

We provide the comprehensive help message for hyperparameter sensitivity testing and ablation study

```
usage: main.py [-h] [--datapath DATAPATH] [--dataset DATASET] [--model MODEL] [--hiddenSize HIDDENSIZE] [--n_factor N_FACTOR] [--epoch EPOCH]
               [--activate ACTIVATE] [--n_sample_all N_SAMPLE_ALL] [--n_sample N_SAMPLE] [--nonhybrid] [--w W] [--gpu_id GPU_ID]
               [--batch_size BATCH_SIZE] [--lr LR] [--lr_dc LR_DC] [--lr_dc_step LR_DC_STEP] [--l2 L2] [--layer LAYER] [--n_iter N_ITER] [--seed SEED]
               [--dropout_gcn DROPOUT_GCN] [--dropout_local DROPOUT_LOCAL] [--dropout_global DROPOUT_GLOBAL] [--e E] [--disen DISEN] [--lamda LAMDA]
               [--norm] [--sw_edge] [--item_edge ITEM_EDGE] [--validation] [--valid_portion VALID_PORTION] [--alpha ALPHA] [--patience PATIENCE]
               [--price_seq PRICE_SEQ] [--trans_counts TRANS_COUNTS] [--nhead NHEAD] [--input_dim INPUT_DIM] [--hidden_dim HIDDEN_DIM]
               [--dropout_tran DROPOUT_TRAN] [--trans TRANS] [--max_seq_length MAX_SEQ_LENGTH] [--inner_size INNER_SIZE]
               [--hidden_dropout_prob HIDDEN_DROPOUT_PROB] [--attn_dropout_prob ATTN_DROPOUT_PROB] [--hidden_act HIDDEN_ACT]
               [--layer_norm_eps LAYER_NORM_EPS] [--initializer_range INITIALIZER_RANGE] [--n_layers N_LAYERS] [--n_heads N_HEADS]
               [--num_layers NUM_LAYERS] [--dropout_prob DROPOUT_PROB] [--hidden_size HIDDEN_SIZE] [--dnn_type DNN_TYPE] [--item_dropout ITEM_DROPOUT]
               [--sess_dropout SESS_DROPOUT] [--temperature TEMPERATURE] [--prototype_size PROTOTYPE_SIZE] [--interest_size INTEREST_SIZE]
               [--tau_ratio TAU_RATIO] [--reg_loss_ratio REG_LOSS_RATIO] [--n_users N_USERS] [--require_pow]

options:
  -h, --help            show this help message and exit
  --datapath DATAPATH   path to the dataset
  --dataset DATASET     popular_20
  --model MODEL         [IHRL,SASRec,GRU4Rec,CORE,SINE,LightGCN]
  --hiddenSize HIDDENSIZE
  --n_factor N_FACTOR   Disentangle factors number
  --epoch EPOCH
  --activate ACTIVATE
  --n_sample_all N_SAMPLE_ALL
  --n_sample N_SAMPLE
  --nonhybrid           only use the global preference to predict
  --w W                 max window size
  --gpu_id GPU_ID
  --batch_size BATCH_SIZE
  --lr LR               learning rate.
  --lr_dc LR_DC         learning rate decay.
  --lr_dc_step LR_DC_STEP
                        the number of steps after which the learning rate decay.
  --l2 L2               l2 penalty
  --layer LAYER         the number of layer used
  --n_iter N_ITER
  --seed SEED
  --dropout_gcn DROPOUT_GCN
                        Dropout rate.
  --dropout_local DROPOUT_LOCAL
                        Dropout rate.
  --dropout_global DROPOUT_GLOBAL
                        Dropout rate.
  --e E                 Disen H sparsity.
  --disen DISEN         use disentangle
  --lamda LAMDA         aux loss weight
  --norm                use norm
  --sw_edge             slide_window_edge
  --item_edge ITEM_EDGE
                        item_edge
  --validation          validation
  --valid_portion VALID_PORTION
                        split the portion
  --alpha ALPHA         Alpha for the leaky_relu.
  --patience PATIENCE
  --price_seq PRICE_SEQ
                        use trans_counts
  --trans_counts TRANS_COUNTS
                        use trans_counts
  --nhead NHEAD         multi_head number
  --input_dim INPUT_DIM
                        input_dim for MA equeal to user_price_length
  --hidden_dim HIDDEN_DIM
                        The inner hidden size in transform layer
  --dropout_tran DROPOUT_TRAN
                        dropout rate
  --trans TRANS         if use transformer for user embedding
  --max_seq_length MAX_SEQ_LENGTH
                        max_seq_length for sequeen
  --inner_size INNER_SIZE
                        The inner hidden size in feed-forward layer
  --hidden_dropout_prob HIDDEN_DROPOUT_PROB
                        The probability of an element to be zeroed
  --attn_dropout_prob ATTN_DROPOUT_PROB
                        The probability of an attention score to be zeroed
  --hidden_act HIDDEN_ACT
  --layer_norm_eps LAYER_NORM_EPS
                        A value added to the denominator for numerical stability.
  --initializer_range INITIALIZER_RANGE
                        The standard deviation for normal initialization.
  --n_layers N_LAYERS   The number of transformer layers in transformer encoder.
  --n_heads N_HEADS     The number of attention heads for multi-head attention layer.
  --num_layers NUM_LAYERS
                        The number of layers in GRU4Rec.
  --dropout_prob DROPOUT_PROB
                        The dropout rate for GRU4Rec.
  --hidden_size HIDDEN_SIZE
                        The inner hidden size in feed-forward layer
  --dnn_type DNN_TYPE
  --item_dropout ITEM_DROPOUT
                        The probability of candidate item embeddings to be zeroed.
  --sess_dropout SESS_DROPOUT
                        The probability of item embeddings in a session to be zeroed.
  --temperature TEMPERATURE
                        Temperature for contrastive loss.
  --prototype_size PROTOTYPE_SIZE
                        The value added to the denominator for numerical stability.
  --interest_size INTEREST_SIZE
                        The number of intentions
  --tau_ratio TAU_RATIO
                        The tau value for temperature tuning
  --reg_loss_ratio REG_LOSS_RATIO
                        The L2 regularization weight.
  --n_users N_USERS     The value added to the denominator for numerical stability.
  --require_pow         The value added to the denominator for numerical stability.
```

