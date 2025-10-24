import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

import torch

from sklearn.metrics import mean_squared_error
from math import sqrt

from stance_scorer import StanceScorer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torchmetrics.text.bert import BERTScore

from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import os
import io
import json
import re
import pickle as pkl
import argparse

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
CHKPT_PATH   = './JointCL/best_checkpoints/vast-zero-shot/state_dict/best_f1_model.bin'
CLUSTER_PATH = './JointCL/best_checkpoints/vast-zero-shot/cluster_result'
VAD_PATH     = './NRC-VAD-Lexicon.txt'
KP_CHKPT     = 'ml6team/keyphrase-generation-t5-small-openkp'
BERT_CHKPT   = 'microsoft/deberta-large-mnli'

wordnet_lemmatizer = WordNetLemmatizer()

def parse_args():
    parser = argparse.ArgumentParser(
        prog = 'PoliSum Perspective Evaluation (stance, object, intensity)',
        description = 'Evaluate a set of model predictions on PoliSum using perspective-centric metrics'
    )
    
    parser.add_argument('-cf', '--config',
                        type = str)
    
    parser.add_argument('-rd', '--results-dir',
                        type = str)
    
    parser.add_argument('-m', '--models',
                        nargs = '+',
                        type = str)
    
    parser.add_argument('-ci', '--conf-interval',
                        type = float,
                        default = 1.96)
    
    parser.add_argument('-pc', '--pred-cols',
                        type = str,
                        nargs = '+',
                        default = ['sp_left_pred', 'sp_right_pred', 'mp_left_pred', 'mp_right_pred'])
    
    parser.add_argument('-rc', '--ref-cols',
                        type = str,
                        nargs = '+',
                        default = ['left_sum', 'right_sum', 'left_sum', 'right_sum'])
    
    parser.add_argument('-sc', '--src-cols',
                        type = str,
                        nargs = '+',
                        default = ['left_op', 'right_op', 'left_op', 'right_op'])
    
    parser.add_argument('-fs', '--first-sentence',
                        action = 'store_true')
    
    parser.add_argument('-f', '--full-pred',
                        action = 'store_false')
    
    parser.add_argument('-r', '--reverse',
                        action = 'store_true')
    
    parser.add_argument('-sd', '--stance-device',
                        default = 0)
    
    parser.add_argument('-kpd', '--kp-device',
                        default = 1)
    
    parser.add_argument('-bd', '--bert-device',
                        default = 3)
    
    parser.set_defaults(first_sentence=True)
    
    args = vars(parser.parse_args())
    
    if args['config'] is not None:
        with open(args['config'], 'r') as f:
            args.update(json.load(f))
            
    return args

def get_devices(args):
    device_dict = {}
    for metric in ('stance', 'kp', 'bert'):
        if args[f'{metric}_device'] == -1:
            device_dict[metric] = torch.device('cpu')
        else:
            device_dict[metric] = torch.device(f'cuda:{args[metric + "_device"]}')
    return device_dict

def setup_eval(args):
    sscorer  = StanceScorer(CHKPT_PATH, CLUSTER_PATH, device = get_devices(args)['stance'])
    vad_dict = load_VAD_lexicons()
    kp_tokenizer = T5Tokenizer.from_pretrained(KP_CHKPT)
    kp_model = T5ForConditionalGeneration.from_pretrained(KP_CHKPT).to(get_devices(args)['kp'])
    bscorer = BERTScore(model_name_or_path = BERT_CHKPT, device = get_devices(args)['bert'])
    
    return {'stance': sscorer,
            'intensity': vad_dict,
            'target': (kp_tokenizer, kp_model),
            'bert': bscorer
           }
    
def setup_data(args):
    model_files = {model: f'{args["results_dir"]}/{model.split("/")[-1]}/preds_bmetrics.csv' for model in args['models']}
    
    all_preds = {model: pd.read_csv(preds) for model, preds in model_files.items()}
    
    return all_preds

# --- INTENSITY ---

def load_VAD_lexicons():
    with open(VAD_PATH, 'r') as infile:
        lines = infile.read()
        lines = lines.split("\n")
        
        vad_dict = {}
        
        for l in lines:            
            lexicon, v_score, a_score, d_score = l.split("\t")

            vad_dict[lexicon] = {
                'v': float(v_score),
                'a': float(a_score),
                'd': float(d_score)
            }
        
        return vad_dict
    
def preprocess_text(text, join_again=True):
    text = text.replace("U.S.", "USA")
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    tokens = [wordnet_lemmatizer.lemmatize(w) for w in word_tokenize(text)]
    
    if join_again:
        text = " ".join(tokens)
        return text
    else:
        return tokens

def get_intensity_score(text, vad_dict):
    instance_tokens = preprocess_text(text).split(' ')
    
    arousals = []
    for t in instance_tokens:
        if t in vad_dict:
            if vad_dict[t]['v'] > 0.65 or vad_dict[t]['v'] < 0.35: # negative
                arousals.append(vad_dict[t]['a'])
    if len(arousals) > 0:
        return sum(arousals)/len(arousals)
    else:
        return 0
            
# --- TARGET ---
def get_keyphrases(text, kp_tok, kp_model, device = torch.device('cuda:0')):
    tok_text = kp_tok(text, return_tensors = 'pt').to(device)
    gen      = kp_model.generate(**tok_text)
    
    keyphrases = kp_tok.batch_decode(gen)[0]
    return keyphrases

def get_target_score(bscorer, source_kp, text_kp):
    return bscorer(preds = text_kp, target = source_kp)['f1']

def eval_row(row, args, scorers, pred_col = 'pred', ref_col = 'ref', src_col = 'alt_op', first_sent = True):
    
    pred, ref, src = row[pred_col], row[ref_col], row[src_col]
    
    if first_sent:
        pred = sent_tokenize(pred)[0]
    
    try:
        stance_score = scorers['stance'].get_stance_score(row['title'], pred, ref)
    except:
        print('Title: ', row['title'])
        print('Pred: ', pred)
        print('Ref: ', ref)
        print('\n')
    
    pred_int = get_intensity_score(pred, scorers['intensity'])
    src_int  = get_intensity_score(src, scorers['intensity'])
    intens_score = (pred_int - src_int)**2
    
    pred_kp = get_keyphrases(pred, scorers['target'][0], scorers['target'][1], device = get_devices(args)['kp'])
    
    
    row[f'{pred_col}_stance'] = stance_score
    row[f'{pred_col}_intens'] = intens_score
    row[f'{pred_col}_kp'] = pred_kp
    
    return row

def eval_df(args, scorers,
            preds, 
            pred_cols, ref_cols, src_cols,
            first_sent = True):
    
    for src_col in src_cols:
        preds[f'{src_col}_kp'] = preds[src_col].apply(lambda src: get_keyphrases(src, scorers['target'][0], scorers['target'][1], device = get_devices(args)['kp']))
    
    for pred_col, ref_col, src_col in zip(pred_cols, ref_cols, src_cols):
        print(f'\tEvaluating {pred_col} against {ref_col}...')
        preds[pred_col] = preds[pred_col].astype(str)
        preds[ref_col] = preds[ref_col].astype(str)
        
        preds = preds.progress_apply(lambda row: eval_row(row, args, scorers, pred_col, ref_col, first_sent = first_sent), axis = 1)
        
        preds[f'{pred_col}_target'] = get_target_score(scorers['bert'], preds[f'{pred_col}_kp'].tolist(), preds[f'{src_col}_kp'].tolist())
    
    return preds

if __name__ == '__main__':
    
    args = parse_args()
    print('Parsed arguments: \n' + "\n".join(['\t' + str(name) + ': ' + str(val) for name, val in args.items()]))
    
    all_preds = setup_data(args)
    print(f'Read all predictions from {args["results_dir"]}')
    
    devices = get_devices(args)
    scorers = setup_eval(args)
       
    print(f'Beginning eval for {len(args["models"])} models')
    
    for model, preds in all_preds.items():
        print(f'\tEvaluating {model}')
        all_preds[model] = eval_df(args, scorers,
                                   all_preds[model],
                                   args['pred_cols'], args['ref_cols'], args['src_cols'],
                                   first_sent = args['first_sentence'])
        
        res_fp = f'{args["results_dir"]}/{model}/base_scores.csv'
        new_res_fp =  f'{args["results_dir"]}/{model}/base_stance_scores.csv'
        pred_fp = f'{args["results_dir"]}/{model}/preds_bsmetrics.csv'
        all_results = pd.read_csv(res_fp)
        stance_scores, stance_cis = [], []
        
        for pred_col in all_results['exp'].values:
            
            score_col = f'{pred_col}_stance'
            avg_score = all_preds[model][score_col].mean()
            ci_score = args['conf_interval']*all_preds[model][score_col].std()/sqrt(len(all_preds[model]))
            stance_scores.append(avg_score*100)
            stance_cis.append(ci_score*100)
        
        all_preds[model].to_csv(pred_fp, index = None)
        
        all_results['stance'] = stance_scores
        all_results['stance_ci'] = stance_cis
        all_results.to_csv(new_res_fp, index = None)
    
    print('Finished evaluation of all models')