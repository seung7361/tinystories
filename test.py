import numpy as np
import torch
import deepspeed
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

dataset = load_dataset("skeskinen/TinyStories-hf")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='[PAD]')
vocab_size = tokenizer.vocab_size

config = GPT2Config(n_embd=768, n_layer=2, n_head=8)
model = GPT2Model.from_pretrained('gpt2', config=config).cuda()
print('Total parameters: {:_}'.format(sum(p.numel() for p in model.parameters())))


a = torch.tensor([50256] * 1024).unsqueeze(0).long().cuda()
b = model(a)['last_hidden_state']
print(b.shape)