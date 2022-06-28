# https://colab.research.google.com/drive/1vnpMoZoenRrWeaxMyfYK4DDbtlBu-M8V?usp=sharing#scrollTo=H1ag9Z0iZbzG
# https://github.com/ivanlai/Conditional_Text_Generation
# https://towardsdatascience.com/conditional-text-generation-by-fine-tuning-gpt-2-11c1a9fc639d

import os
import io
import requests
import numpy as np
import pandas as pd
import re
import zipfile
import random
import time
import csv
import datetime
from itertools import compress
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from process_data import get_dataset_for_gpt2

from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
                         AdamW, get_linear_schedule_with_warmup, \
                         TrainingArguments, BeamScorer, Trainer

import torch
from torch.utils.data import Dataset, random_split, DataLoader, \
                             RandomSampler, SequentialSampler

from IPython.display import clear_output

print(f"PyTorch version: {torch.__version__}")

DEBUG           = False

INPUT_DIR       = 'articles'

USE_APEX        = True
APEX_OPT_LEVEL  = 'O1'

MODEL           = 'gpt2' #{gpt2, gpt2-medium, gpt2-large, gpt2-xl}

UNFREEZE_LAST_N = 6 #The last N layers to unfreeze for training

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
                    
MAXLEN          = 768  #{768, 1024, 1280, 1600}

TRAIN_SIZE      = 0.8

if USE_APEX:
    TRAIN_BATCHSIZE = 16
    BATCH_UPDATE    = 16
else:
    TRAIN_BATCHSIZE = 8
    BATCH_UPDATE    = 32

EPOCHS          = 15
LR              = 5e-4
EPS             = 1e-8
WARMUP_STEPS    = 1e2

SEED            = 2020


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


class myDataset(Dataset):

    def __init__(self, data, tokenizer, randomize=True):

        title, text = [], []
        for k, v in data.items():
            title.append(v[0])
            text.append(v[1])

        self.randomize = randomize
        self.tokenizer = tokenizer 
        self.title     = title
        self.text      = text

    #---------------------------------------------#

    def __len__(self):
        return len(self.text)

    #---------------------------------------------#
    
    def __getitem__(self, i):
        
        input = SPECIAL_TOKENS['bos_token'] + self.title[i] + \
                SPECIAL_TOKENS['sep_token'] + \
                self.text[i] + SPECIAL_TOKENS['eos_token']

        encodings_dict = tokenizer(input,                                   
                                   truncation=True, 
                                   max_length=MAXLEN, 
                                   padding="max_length")   
        
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']
        
        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids), 
                'attention_mask': torch.tensor(attention_mask)}


def split_data(data, S=TRAIN_SIZE):
    # Shuffle ids
    ids = list(data.keys())
    random.shuffle(ids)

    # Split into training and validation sets    
    train_size = int(S * len(data))

    train_ids = ids[:train_size]
    val_ids = ids[train_size:]

    train_data = dict()
    for id in train_ids:
        train_data[id] = data[id]

    val_data = dict()
    for id in val_ids:
        val_data[id] = data[id]

    return train_data, val_data




def get_tokenizer(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer




def get_model(tokenizer, special_tokens=None, load_model_path=None):

    #GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL, 
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else: 
        config = AutoConfig.from_pretrained(MODEL,                                     
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)    

    #----------------------------------------------------------------#
    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

    if special_tokens:
        #Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    model.cuda()
    return model


tokenizer = get_tokenizer(special_tokens=SPECIAL_TOKENS)
model = get_model(tokenizer, 
                  special_tokens=SPECIAL_TOKENS,
                #   load_model_path='pytorch_model.bin'
                 )


# - Freeze selective layers:
# - Freeze all layers except last n:
for parameter in model.parameters():
    parameter.requires_grad = False

for i, m in enumerate(model.transformer.h):        
    #Only un-freeze the last n transformer blocks
    if i+1 > 12 - UNFREEZE_LAST_N:
        for parameter in m.parameters():
            parameter.requires_grad = True 

for parameter in model.transformer.ln_f.parameters():        
    parameter.requires_grad = True

for parameter in model.lm_head.parameters():        
    parameter.requires_grad = True





data = get_dataset_for_gpt2()


train_data, val_data = split_data(data)

train_dataset = myDataset(train_data, tokenizer)
val_dataset = myDataset(val_data, tokenizer, randomize=False)

print( f'There are {len(train_dataset) :,} samples for training, and {len(val_dataset) :,} samples for validation testing')

training_args = TrainingArguments(
    output_dir="gpt2-data/",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCHSIZE,
    per_device_eval_batch_size=TRAIN_BATCHSIZE,
    gradient_accumulation_steps=BATCH_UPDATE,
    evaluation_strategy="epoch",
    fp16=True,
    fp16_opt_level=APEX_OPT_LEVEL,
    warmup_steps=WARMUP_STEPS,    
    learning_rate=LR,
    adam_epsilon=EPS,
    weight_decay=0.01,        
    save_total_limit=1,
    load_best_model_at_end=True,    
     
)

#---------------------------------------------------#
trainer = Trainer(
    model=model,
    args=training_args,    
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

#---------------------------------------------------#
# trainer.train()
# trainer.save_model()  