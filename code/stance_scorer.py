# Adapted from "JointCL: A Joint Contrastive Learning Framework for Zero-Shot Stance Detection": https://github.com/HITSZ-HLT/JointCL

import pickle as pkl
import io
import os
import torch
import numpy as np
from transformers import BertModel
from jointcl_models.bert_scl_prototype_graph import BERT_SCL_Proto_Graph
from jointcl_utils.data_utils import Tokenizer4Bert
from argparse import Namespace
from sklearn.metrics import mean_squared_error
from math import sqrt

class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
    
def rmse(num1, num2):
    if isinstance(num1, float) and isinstance(num2, float):
        return sqrt(mean_squared_error([num1], [num2]))
    else:
        return sqrt(mean_squared_error(num1, num2))
            
def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

class StanceScorer:
    OPT = Namespace(model_name='bert-scl-prototype-graph', 
                    type=0, 
                    dataset='zeroshot', 
                    output_par_dir='../best_checkpoints/vast-zero-shot', 
                    polarities=['pro', 'con', 'neutral'], 
                    optimizer='adam', 
                    temperature=0.07, 
                    initializer='xavier_uniform_', 
                    lr=5e-06, dropout=0.1, 
                    l2reg=0.001, 
                    log_step=10, 
                    log_path='./log', 
                    embed_dim=300, 
                    hidden_dim=128, feature_dim=256, 
                    output_dim=64, relation_dim=100, bert_dim=768, 
                    pretrained_bert_name='bert-base-uncased', 
                    max_seq_len=200, 
                    test_dir='../datasets/VAST/vast_test.csv', 
                    stance_loss_weight=1, prototype_loss_weight=0.01,
                    alpha=0.8, beta=1.2, 
                    device=torch.device('cuda:0'),
                    seed=666, 
                    batch_size=16, 
                    eval_batch_size=16, 
                    cluster_times=5, 
                    gnn_dims='192,192', 
                    att_heads='4,4', 
                    dp=0.1, 
                    n_gpus=1, 
                    model_class=BERT_SCL_Proto_Graph, optim_class=torch.optim.Adam, 
                    input_features=['concat_bert_indices', 'concat_segments_indices'], 
                    output_dir='../best_checkpoints/vast-zero-shot/bert-scl-prototype-graph/zeroshot/2024-01-31 13-13-01', 
                    num_labels=3)
    def __init__(self, chkpt_path, cluster_path, device = torch.device('cuda:0')):
        self.device = device
        StanceScorer.OPT.device = device

        # Load Proto models
        self.bert_proto = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.tokenizer = Tokenizer4Bert(256, 'bert-base-uncased')

        # Load stance model
        self.stance_model = BERT_SCL_Proto_Graph(StanceScorer.OPT, self.bert_proto).to(self.device)
        model_chkpt = torch.load(chkpt_path, map_location = self.device)
        self.stance_model.load_state_dict(model_chkpt)

        # Load cluster results
        self.cluster_result = CPU_Unpickler(open(cluster_path, 'rb')).load()

    
    def get_stance(self, aspect, text):
    
        # Calculate Features
        text_indices = self.tokenizer.text_to_sequence(text)
        aspect_indices = self.tokenizer.text_to_sequence(aspect, max_seq_len=4)
        text_len = np.sum(text_indices != 0)
        aspect_len = np.sum(aspect_indices != 0)
        concat_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + aspect + " [SEP]")
        concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
        concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)
    
        # Tensorize Features
        concat_bert_indices = torch.tensor(concat_bert_indices).unsqueeze(0).to(self.device)
        concat_segments_indices = torch.tensor(concat_segments_indices).unsqueeze(0).to(self.device)
        input_features = [concat_bert_indices, concat_segments_indices]
    
        # Predict
        logits, _ = self.stance_model(input_features + self.cluster_result)
        probs = logits.softmax(dim = -1)
    
        return probs
    
    def get_stance_score(self, target, pred, ref):
        pred_stance = self.get_stance(target, pred)
        ref_stance  = self.get_stance(target, ref)

        stance_label = ref_stance.argmax()
        return rmse(pred_stance[0][stance_label].item(), ref_stance[0][stance_label].item())