from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2LMHeadModel,AdamW
import numpy as np 
import torch

# sample data provided by the patient and doctor
# sampleData = pd.read_csv('') # add dataset

# From dataset fine tune the model
class MNDDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        with open(file_path, encoding="utf-8") as f:
            self.examples = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(
            self.examples[idx],
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0]
        }

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

train_dataset = MNDDataset('Model\mnd_facts.txt', tokenizer)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    print(f"Starting epoch {epoch+1}")
    for i, batch in enumerate(train_loader):
        print(f"Processing batch {i+1} of {len(train_loader)}")
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()

        # Shift the inputs to the right
        inputs = {
            'input_ids': input_ids[:, :-1],
            'attention_mask': attention_mask[:, :-1]
        }
        labels = labels[:, 1:]
        
        optimizer.zero_grad()
        loss = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)[0]
        loss.backward()
        optimizer.step()

model.save_pretrained('./mnd_facts_model')
