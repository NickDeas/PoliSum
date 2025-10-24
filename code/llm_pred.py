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

import torch
from torch.utils.data import DataLoader

from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, MistralForCausalLM
from datasets import Dataset, load_dataset

torch.manual_seed(0)
np.random.seed(0)

MP_L_INST = 'Produce a short, single-sentence summary of the left-leaning political perspective within the following texts. Begin the summary with "The left": '
MP_R_INST = 'Produce a short, single-sentence summary of the right-leaning political perspective within the following texts. Begin the summary with "The right": '
SP_L_INST = 'Produce a short, single-sentence summary of the left-leaning political perspective within the following texts. Begin the summary with "The left": '
SP_R_INST = 'Produce a short, single-sentence summary of the right-leaning political perspective within the following texts. Begin the summary with "The right": '

CACHE_DIR = '<CACHE>' #Replace with local cache directory
HF_LOSS_IGNORE_TOKEN_ID = -100

def parse_args():
    
    parser = argparse.ArgumentParser(
        prog = 'LLM Polisum Eval',
        description = 'Test a Mistral, Mixtral, or Llama model on PoliSum'
    )
    
    parser.add_argument('-cf', '--config',
                        type = str)
    
    parser.add_argument('-fp', '--file-path',
                        type = str)
    
    parser.add_argument('-rd', '--results-dir',
                        type = str)
    
    parser.add_argument('-m', '--model',
                        type = str)

    parser.add_argument('-dt', '--dtype',
                        type = str,
                        default = 'float16')

    parser.add_argument('-lc', '--lora-checkpoint',
                        type = str,
                        default = '')

    parser.add_argument('-fa', '--flash-attn',
                        type = bool,
                        default = True)
    
    parser.add_argument('-ac', '--activation-checkpointing',
                        type = bool,
                        default = True)
    
    parser.add_argument('-c', '--compile',
                        type = bool,
                        default = False)
    
    parser.add_argument('-sp', '--single',
                        type = bool,
                        default = True)
    
    parser.add_argument('-mp', '--mixed',
                        type = bool,
                        default = True)
    
    parser.add_argument('-d', '--device',
                        type = int,
                        default = -1)
    
    args = vars(parser.parse_args())
    
    if args['config'] is not None:
        with open(args['config'], 'r') as f:
            args.update(json.load(f))
            
    return args

def setup_data(args):
    
    data = pd.read_csv(args['file_path'])
    
    return data
    
def get_device(args):
    if args['device'] == -1:
        return torch.device('cpu')
    elif args['device'] == -2:
        return 'auto'
    else:
        return torch.device(f'cuda:{args["device"]}')

def setup_model(args):

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args['model'], cache_dir = CACHE_DIR)
    if 'mistral' in args['model']:
        tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Finished initializing tokenizer")
    # Initialize the model and optimizer
    # Figure out data type
    if args['dtype'] == "float16":
        torch_dtype = torch.float16
    elif args['dtype'] == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args['dtype'] == "float32":
        torch_dtype = torch.float32
        
    
    model = AutoModelForCausalLM.from_pretrained(args['model'], device_map=get_device(args), torch_dtype=torch_dtype, use_flash_attention_2=args['flash_attn'], cache_dir = CACHE_DIR)

    model = torch.compile(model, disable=not args['compile'])
    print(f"Finished loading model and lora")
    
    return model, tokenizer
    
def get_sum_pred(args, text, model, tok, device, exp):
    if exp == 'left_mp':
        inst = MP_L_INST
    elif exp == 'right_mp':
        inst = MP_R_INST
    elif exp == 'left_sp':
        inst = SP_L_INST
    elif exp == 'right_sp':
        inst = SP_R_INST
    
    if 'mistral' in args['model']:
        inputs = f'<s>[INST]\n{inst}\n{text}\n[/INST]\n'
        tok_text = tok([inputs], return_tensors = 'pt').to(device)
        gen = model.generate(tok_text['input_ids'], 
                             attention_mask = tok_text['attention_mask'],
                             max_new_tokens = 128, 
                             pad_token_id = tok.eos_token_id)
        gen_text = tok.batch_decode(gen, skip_special_tokens = True)[0]
    elif 'vicuna' in args['model']:
        inputs = f'USER: {inst}\n{text}\nASSISTANT:'
        tok_text = tok([inputs], return_tensors = 'pt').to(device)
        gen = model.generate(tok_text['input_ids'], 
                             attention_mask = tok_text['attention_mask'],
                             max_new_tokens = 128, 
                             pad_token_id = tok.eos_token_id)
        gen_text = tok.batch_decode(gen, skip_special_tokens = True)[0]
    elif 'mixtral' in args['model'].lower() or 'llama' in args['model'].lower():
        messages = [
            {'role': 'user', 'content': f'{inst}\n{text}'} 
        ]

        tok_text = tok.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors = 'pt'
        ).to(device)
        
        terminators = [
            tok.eos_token_id,
            tok.convert_tokens_to_ids("<|eot_id|>")
        ]
        gen = model.generate(tok_text, 
                             do_sample = False,
                             top_p = None,
                             temperature = None,
                             max_new_tokens = 64, 
                             eos_token_id = terminators,
                             pad_token_id = tok.pad_token_id)
        gen = gen[0][tok_text.shape[-1]:]
        gen_text = tok.decode(gen, skip_special_tokens = True)
        # print(gen_text)
    if 'mistral' in  args['model']:
        gen_text = gen_text[gen_text.find('[/INST]') + 8:]
    elif 'vicuna' in args['model']:
        gen_text = gen_text[gen_text.find('ASSISTANT:') + 11:-3]
    elif 'llama' in args['model']:
        if gen_text[:4].lower() == 'here':
            gen_text = gen_text[gen_text.find(':') + 1:]
        
    return gen_text

if __name__ == '__main__':
    
    args = parse_args()
    print('Parsed arguments: \n' + "\n".join(['\t' + str(name) + ': ' + str(val) for name, val in args.items()]))
    
    data = setup_data(args)
    print(f'Read Data from {args["file_path"]}')
        
    dt_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint = args['model']
    while checkpoint[-1] == '/':
        checkpoint = checkpoint[:-1]
    res_dir = f'{args["results_dir"]}/{checkpoint.split("/")[-1]}/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    print(f'Beginning eval for {args["model"]}')
    results = defaultdict(list)

    model, tok = setup_model(args)
    device = model.device
    
    if args['mixed']:
        data[f'mp_left_pred'] = data['alt_op'].progress_apply(lambda text: get_sum_pred(args, text, model, tok, device, exp = 'left_mp'))
        data.to_csv(res_dir + '/preds.csv', index = None)
        print(f'Saved predictions to {res_dir}') 
    
        data[f'mp_right_pred'] = data['alt_op'].progress_apply(lambda text: get_sum_pred(args, text, model, tok, device, exp = 'right_mp'))

        data.to_csv(res_dir + '/preds.csv', index = None)
        print(f'Saved predictions to {res_dir}') 
    
    if args['single']:
        data[f'sp_left_pred'] = data['left_op'].progress_apply(lambda text: get_sum_pred(args, text, model, tok, device, exp = 'left_sp'))
        data.to_csv(res_dir + '/preds.csv', index = None)
        print(f'Saved predictions to {res_dir}') 

        data[f'sp_right_pred'] = data['right_op'].progress_apply(lambda text: get_sum_pred(args, text, model, tok, device, exp = 'right_sp'))
        data.to_csv(res_dir + '/preds.csv', index = None)
        print(f'Saved predictions to {res_dir}')
    
    print(f'Finished inference with {args["model"]}')
        
        
        
        
        
        
    
    
    
    
    
