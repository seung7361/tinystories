import numpy as np
import torch
import deepspeed
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

dataset = load_dataset("skeskinen/TinyStories-hf")
deepspeed.init_distributed()

### hyperparameters for training

num_epochs = 5
learning_rate = 5e-5
warmup_steps = 5000
batch_size = 32

n_embd = 768
n_layer = 2
n_head = 8
max_length = 1024

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
vocab_size = tokenizer.vocab_size + 1

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
        input_ids = tokenizer(text, return_tensors='pt')['input_ids'].cuda()
        
        for i in range(max_length):
            outputs = self(input_ids)
            next_token_logits = outputs[0][-1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            if next_token_id == 50256 or next_token_id == 50257:
                break
        
        return tokenizer.decode(input_ids[0])

model = GPT2ModelLM().cuda()

print('Total parameters: {:_}'.format(sum(p.numel() for p in model.parameters())))

train_dataset = torch.load('train_dataset.pt')
val_dataset = torch.load('val_dataset.pt')
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=50257)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=(batch_size)*num_epochs)

### training loop

model_engine, optimizer_engine, train_loader, _ = deepspeed.initialize(model=model,
                                                                         optimizer=optimizer,
                                                                         model_parameters=model.parameters(),
                                                                         config='ds_config.json',
                                                                         training_data=train_dataset,

                                                                         lr_scheduler=scheduler)


for epoch in range(num_epochs):
    model_engine.train()

    # save checkpoint
    if epoch % 5 == 0:
        torch.save(model.state_dict(), 'model_epoch_{}.pt'.format(epoch))
        print('saving checkpoint complete.')

    for batch in tqdm(train_loader):
        model_engine.zero_grad()
        batch = batch.cuda()

        input_ids, labels = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        outputs = model_engine(input_ids)

        loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
        model_engine.backward(loss)
        model_engine.step()
    
    model_engine.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            batch = batch.cuda()

            input_ids, labels = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            outputs = model_engine(input_ids)

            loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
            total_loss += loss.item()

    print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, total_loss/len(val_dataloader)))

### save model

torch.save(model.state_dict(), 'model.pt')
print('saving model complete.')