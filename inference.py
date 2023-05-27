import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config


config = GPT2Config()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

num_epochs = 5
learning_rate = 5e-5
warmup_steps = 5000
batch_size = 32
vocab_size = tokenizer.vocab_size + 1

n_embd = 768
n_layer = 2
n_head = 8
max_length = 1024

config = GPT2Config(n_embd=n_embd, n_layer=n_layer, n_head=n_head, vocab_size=vocab_size)
gpt2model = GPT2Model.from_pretrained('gpt2', config=config, ignore_mismatched_sizes=True).cuda()


class GPT2ModelLM(torch.nn.Module):
    def __init__(self, gpt2model=gpt2model, n_embd=n_embd, vocab_size=vocab_size):
        super().__init__()
        self.gpt2model = gpt2model
        self.linear = torch.nn.Linear(n_embd, vocab_size)
    
    def forward(self, input_ids):
        outputs = self.gpt2model(input_ids)['last_hidden_state']
        outputs = self.linear(outputs)

        return outputs
    
    def generate(self, text, max_length=1024):
        with torch.no_grad():
            input_ids = tokenizer(text, return_tensors='pt')['input_ids'].cuda()
            for i in range(max_length):
                outputs = self(input_ids)
                next_token_logits = outputs[0][-1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                input_ids = torch.cat([input_ids.squeeze(0), next_token_id], dim=-1).unsqueeze(0)
                if next_token_id == 50256 or next_token_id == 50257:
                    break
            
            return tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
print('model init complete.')
model = GPT2ModelLM().cuda()
model.load_state_dict(torch.load('model.pt'))
print('model load complete.')

print(model.generate('Once upon a time', max_length=256))