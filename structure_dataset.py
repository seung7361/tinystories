import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

dataset = load_dataset("skeskinen/TinyStories-hf")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='[PAD]')
vocab_size = tokenizer.vocab_size
print(tokenizer.pad_token_id)

train_dataset = [tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=1024, truncation=True)['input_ids'].long().squeeze(0) for sentence in tqdm(dataset['train']['text'])]
val_dataset = [tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=1024, truncation=True)['input_ids'].long().squeeze(0) for sentence in tqdm(dataset['validation']['text'])]

torch.save(train_dataset, 'train_dataset.pt')
torch.save(val_dataset, 'val_dataset.pt')
print('Datasets saved')