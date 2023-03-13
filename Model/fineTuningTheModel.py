import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, TFGPT2Model
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils import prune
import tensorflow as tf


data = np.loadtxt('datasets\YNMDataset.txt', delimiter=',', dtype=str)

# Define a custom dataset class
class YesNoMaybeDataset(Dataset):
    def __init__(self, questions, labels, tokenizer, max_length):
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(), 'labels': torch.tensor(label)}
        
# Define the hyperparameters
batch_size = 8
learning_rate = 5e-5
num_epochs = 3
max_length = 64

# Define the questions and their corresponding labels
questions = data[1:, 0]
labels = data[1:, 1].astype(int)

# Initialize the tokenizer and the model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = TFGPT2Model.from_pretrained('gpt2-medium')

# Add a classification head on top of the model
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = len(tokenizer)
model.config.classifier_dropout = 0.3
model.config.num_labels = 2

# Define the optimizer and the loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Split the dataset into training and validation sets
split = int(0.8 * len(questions))
train_questions, train_labels = questions[:split], labels[:split]
val_questions, val_labels = questions[split:], labels[split:]

# Create the data loaders
train_dataset = YesNoMaybeDataset(train_questions, train_labels, tokenizer, max_length)
val_dataset = YesNoMaybeDataset(val_questions, val_labels, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device) # Changed this line to move the model to the device

# Train the model
for epoch in range(num_epochs):
    model.train()
    train_loss = []
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

# Prune the model
prune.global_unstructured(
    parameters=model.parameters(),
    pruning_method=prune.L1Unstructured,
    amount=0.9,
)

# Save the pruned model
model.save_pretrained('./question_identifier_model')
