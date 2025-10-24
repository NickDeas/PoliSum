import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

import os
import re
import json
import argparse
from collections import defaultdict
from datetime import datetime
import requests

API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

MP_L_INST = 'Produce a short, single-sentence summary of the left-leaning political perspective within the following texts. Respond only with the summary beginning with "The left": '
MP_R_INST = 'Produce a short, single-sentence summary of the right-leaning political perspective within the following texts. Respond only with the summary beginning with "The right": '
SP_L_INST = 'Produce a short, single-sentence summary of the left-leaning political perspective within the following texts. Respond only with the summary beginning with "The left": '
SP_R_INST = 'Produce a short, single-sentence summary of the right-leaning political perspective within the following texts. Respond only with the summary beginning with "The right": '

def parse_args():
    
    parser = argparse.ArgumentParser(
        prog = 'API Polisum Eval',
        description = 'Test an API-exposed LLM on Polisum'
    )
    
    parser.add_argument('-cf', '--config',
                        type = str)
    
    parser.add_argument('-fp', '--file-path',
                        type = str)
    
    parser.add_argument('-rd', '--results-dir',
                        type = str)
    
    parser.add_argument('-m', '--model',
                        type = str)

    parser.add_argument('-ak', '--api-key',
                        type = str)
    
    parser.add_argument('-en', '--exp-name',
                        type = str)
    
    parser.add_argument('-ss', '--summaries',
                        nargs = '+',
                        type = str,
                        default = ['r_mp', 'l_mp', 'r_sp', 'r_mp'])
    
    args = vars(parser.parse_args())
    
    if args['config'] is not None:
        with open(args['config'], 'r') as f:
            args.update(json.load(f))
            
    return args

def setup_data(args):
    
    data = pd.read_csv(args['file_path'])
    
    return data
    
def get_sum_pred(args, text, exp = 'sp_left'):
    if exp == 'left_mp':
        inst = MP_L_INST
    elif exp == 'right_mp':
        inst = MP_R_INST
    elif exp == 'left_sp':
        inst = SP_L_INST
    elif exp == 'right_sp':
        inst = SP_R_INST
        
    messages = [{'role': 'user', 'content': f'{inst}\n{text}'}]
    
    try:
        payload = {
          "model": f"accounts/fireworks/models/{args['model']}",
          "max_tokens": 128,
          "top_p": 1,
          "top_k": 1,
          "presence_penalty": 0,
          "frequency_penalty": 0,
          "temperature": 0,
          "messages": messages,
          "stop": ['.'],
          "context_length_exceeded_behavior": "truncate",
          "user": args['exp_name'],
        }
        headers = {
          "Accept": "application/json",
          "Content-Type": "application/json",
          "Authorization": f"Bearer {args['api_key']}"
        }
        resp = requests.request("POST", API_URL, headers=headers, data=json.dumps(payload))
        return resp.content
    except Exception as e:
        print('In generate: ', e)
        return 'Error'
    
def extract_sum(resp):
    try:
        resp = json.loads(resp)['choices'][0]['message']['content']
        return resp
    except Exception as e:
        print('In extract: ', e)
        return 'Error'

if __name__ == '__main__':
    
    args = parse_args()
    print('Parsed arguments: \n' + "\n".join(['\t' + str(name) + ': ' + str(val) for name, val in args.items()]))
    
    data = setup_data(args)
    print(f'Read Data from {args["file_path"]}')
        
    checkpoint = args['model']
    while checkpoint[-1] == '/':
        checkpoint = checkpoint[:-1]
    res_dir = f'{args["results_dir"]}/{checkpoint.split("/")[-1]}/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    print(f'Beginning eval for {args["model"]}')
    results = defaultdict(list)
    
    # Mixed Perspective Left
    if 'l_mp' in args['summaries']:
        data[f'mp_left_pred_full'] = data['alt_op'].progress_apply(lambda text: get_sum_pred(args, text, exp = 'left_mp'))
        data[f'mp_left_pred'] = data['mp_left_pred_full'].apply(extract_sum)
        data.to_csv(res_dir + '/preds.csv', index = None)
        print(f'Saved partial predictions to {res_dir}') 
    
    # Mixed Perspective Right
    if 'r_mp' in args['summaries']:
        data[f'mp_right_pred_full'] = data['alt_op'].progress_apply(lambda text: get_sum_pred(args, text, exp = 'right_mp'))
        data[f'mp_right_pred'] = data['mp_right_pred_full'].apply(extract_sum)
        data.to_csv(res_dir + '/preds.csv', index = None)
        print(f'Saved partial predictions to {res_dir}') 
    
    # Single Perspective Left
    if 'l_sp' in args['summaries']:
        data[f'sp_left_pred_full'] = data['left_op'].progress_apply(lambda text: get_sum_pred(args, text, exp = 'left_sp'))
        data[f'sp_left_pred'] = data['sp_left_pred_full'].apply(extract_sum)
        data.to_csv(res_dir + '/preds.csv', index = None)
        print(f'Saved partial predictions to {res_dir}') 
    
    # Single Perspective Right
    if 'r_sp' in args['summaries']:
        data[f'sp_right_pred_full'] = data['right_op'].progress_apply(lambda text: get_sum_pred(args, text, exp = 'right_sp'))
        data[f'sp_right_pred'] = data['sp_right_pred_full'].apply(extract_sum)
        data.to_csv(res_dir + '/preds.csv', index = None)
        print(f'Saved predictions to {res_dir}')
    
    print(f'Finished inference with {args["model"]}')
        
        
        
        
        
        
    
    
    
    
    
