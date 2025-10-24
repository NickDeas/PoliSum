import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
tqdm.pandas()

from nltk import sent_tokenize

import datasets
import evaluate

from torchmetrics.text.bert import BERTScore
from rouge_score import rouge_scorer
from summac.model_summac import SummaCZS, SummaCConv
from alignscore import AlignScore

from collections import defaultdict
import math
from ast import literal_eval
import json
import os

import argparse


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# nlp = spacy.load('en_core_web_sm')

def parse_args():
    parser = argparse.ArgumentParser(
        prog = 'PoliSum Base Evaluation (coverage and faithfulness)',
        description = 'Evaluate a set of model predictions on PoliSum using summary coverage and faithfulness metrics'
    )
    
    parser.add_argument('-cf', '--config',
                        type = str)
    
    parser.add_argument('-rd', '--results-dir',
                        type = str)
    
    parser.add_argument('-m', '--models',
                        nargs = '+',
                        type = str)
    
    parser.add_argument('-bd', '--bert-device',
                        type = int,
                        default = 0)
    
    parser.add_argument('-ad', '--align-device',
                        type = int,
                        default = 0)
    
    parser.add_argument('-sd', '--summac-device',
                        type = int,
                        default = 0)
    
    parser.add_argument('-bc', '--bert-score-chkpt',
                        type = str,
                        default = 'microsoft/deberta-large-mnli')
    
    parser.add_argument('-ac', '--align-chkpt',
                        type = str,
                        default = '/path/to/AlignScore-large.ckpt')
    
    parser.add_argument('-blc', '--bleurt-chkpt',
                        type = str,
                        default = 'BLEURT-20-D6')
    
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
    
    parser.add_argument('-mt', '--metrics',
                        type = str,
                        nargs = '+',
                        default = ['rouge', 'bert', 'bleurt', 'summac', 'menli', 'align'])
    
    parser.set_defaults(first_sentence=True)
    
    args = vars(parser.parse_args())
    
    if args['config'] is not None:
        with open(args['config'], 'r') as f:
            args.update(json.load(f))
            
    return args

def get_devices(args):
    
    devices = {}
    
    for metric in ['bert', 'summac', 'align']:
        if args[f'{metric}_device'] != -1:
            devices[metric] = torch.device(f'cuda:{args[f"{metric}_device"]}')
        else:
            devices[metric] = torch.device('cpu')    
            
    return devices

def setup_eval(args, devices):
    
    metrics = {}
    
    if 'rouge' in args['metrics']:
        metrics['rouge']   = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    if 'bert' in args['metrics']:
        metrics['bert']   = BERTScore(model_name_or_path = args['bert_score_chkpt'], device = devices['bert'])

    if 'summac' in args['metrics']:
        metrics['summac'] = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=devices['summac'], start_file="default", agg="mean")
    
    if 'align' in args['metrics']:
        metrics['align'] = AlignScore(model='roberta-base', batch_size=32, device=devices['align'], ckpt_path=args["align_chkpt"], evaluation_mode='nli_sp', verbose = False)
    
    if 'bleurt' in args['metrics']:
        metrics['bleurt'] = evaluate.load('bleurt', args['bleurt_chkpt'])
    
    return metrics

def eval_summac(metrics, pred, src):
    return metrics['summac'].score([src], [pred])['scores'][0]

def eval_alignscore(metrics, pred, src):
    return metrics['align'].score(contexts=[src], claims = [pred])[0]

def eval_bleurt(metrics, pred, ref):
    return metrics['bleurt'].compute(predictions = [pred], references = [ref])['scores'][0]

def eval_rouge(metrics, pred, ref):
    return metrics['rouge'].score(ref, pred)

def eval_full_row(row, metrics, pred_col = 'pred', ref_col = 'ref', src_col = 'src', first_sent = True, cl_trunc = False):
    pred, ref, src = row[pred_col], row[ref_col], row[src_col]
    
    if cl_trunc:
        pred = clause_truncate(pred, len(ref.split(' ')))
    elif first_sent:
        pred = sent_tokenize(pred)[0]
            
    if isinstance(pred, str) and isinstance(ref, str):
        
        if 'rouge' in metrics.keys():
            rouge = eval_rouge(metrics, pred, ref)
            row[f'{pred_col}_rouge1']   = rouge['rouge1'].fmeasure
            row[f'{pred_col}_rouge2']   = rouge['rouge2'].fmeasure
            row[f'{pred_col}_rougeL']   = rouge['rougeL'].fmeasure
        
        if 'bleurt' in metrics.keys():
            bleurt_score = eval_bleurt(metrics, pred, ref)
            row[f'{pred_col}_bleurt']   = bleurt_score
            
        if 'summac' in metrics.keys():
            summac = eval_summac(metrics, pred, src)
            row[f'{pred_col}_summac']   = summac
        
        if 'align' in metrics.keys():
            ascore = eval_alignscore(metrics, pred, src)
            row[f'{pred_col}_align']    = ascore

    else:        
        if 'rouge' in metrics.keys():
            row[f'{pred_col}_rouge1']   = -1
            row[f'{pred_col}_rouge2']   = -1
            row[f'{pred_col}_rougeL']   = -1
            
        if 'bleurt' in metrics.keys():
            row[f'{pred_col}_bleurt']   = bleurt_score

        if 'summac' in metrics.keys():
            row[f'{pred_col}_summac']   = -1

        if 'align' in metrics.keys():
            row[f'{pred_col}_align']    = -1
        
    return row

def eval_bscore(df, metrics, pred_col, ref_col, first_sent = True):
    if first_sent:
        preds = df[pred_col].apply(lambda t: sent_tokenize(t)[0])
        preds = preds.astype(str).values.tolist()
    else:
        preds = df[pred_col].astype(str).values.tolist()
    bscores = metrics['bert'](preds = preds, target = df[ref_col].astype(str).values.tolist())
    return bscores['f1']

def eval_df(args, metrics, df, pred_cols, ref_cols, src_cols, first_sent = True):

    for pred_col, ref_col, src_col in zip(pred_cols, ref_cols, src_cols):
        print(f'\tEvaluating {pred_col} against {ref_col}...')
        df = df.progress_apply(lambda row: eval_full_row(row, metrics, pred_col, ref_col, src_col), axis = 1)
        print(f'\tFinished Base Metrics: {", ".join(args["metrics"])}')
        if 'bert' in metrics:
            df[f'{pred_col}_bert'] = eval_bscore(df, metrics, pred_col, ref_col)
            print(f'\tFinished BERTScore')
    return df


def setup_data(args):
    
    model_files = {model: f'{args["results_dir"]}/{model}/preds.csv' for model in args['models']}
    
    all_preds = {model: pd.read_csv(preds) for model, preds in model_files.items()}
    
    return all_preds
    
if __name__ == '__main__':
    
    args = parse_args()
    print('Parsed arguments: \n' + "\n".join(['\t' + str(name) + ': ' + str(val) for name, val in args.items()]))
    
    all_preds = setup_data(args)
    print(f'Read all predictions from {args["results_dir"]}')
    
    devices = get_devices(args)
    metrics = setup_eval(args, devices)
    
    metric_names = [met for met in metrics if met != 'rouge']
    if 'rouge' in metrics:
        metric_names = metric_names + ['rouge1', 'rouge2', 'rougeL']
    
    print(f'Beginning eval for {len(args["models"])} models')
    
    for model, preds in all_preds.items():
        print(f'\tEvaluating {model}')
        all_preds[model] = all_preds[model].astype(str)
        all_preds[model] = eval_df(args, metrics, 
                                   all_preds[model],
                                   args['pred_cols'], args['ref_cols'], args['src_cols'], 
                                   first_sent = args['first_sentence'])
        
        all_results = defaultdict(list)
        for pred_col in args['pred_cols']:
            all_results['model'].append(model)
            all_results['exp'].append(pred_col)
            for metric in metric_names:
                score_col = f'{pred_col}_{metric}'
                avg_score = all_preds[model][score_col].mean()
                ci_score = args['conf_interval']*all_preds[model][score_col].std()/math.sqrt(len(all_preds[model]))
                all_results[metric].append(avg_score*100)
                all_results[f'{metric}_ci'].append(ci_score*100)
        
        res_dir = f'{args["results_dir"]}/{model}/'
        all_preds[model].to_csv(f'{res_dir}/preds{"_rev" if args["reverse"] else ""}{"_full" if not args["first_sentence"] else ""}_bmetrics.csv', index = None)
        pd.DataFrame(all_results).to_csv(f'{res_dir}/{"_rev" if args["reverse"] else ""}{"_full" if not args["first_sentence"] else ""}base_scores.csv', index = None)
    
    print('Finished evaluation of all models')
    
    
        
        
        
    
    