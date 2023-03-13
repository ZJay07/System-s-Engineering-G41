import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the fine-tuned model
# fine_tuned_model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
fine_tuned_model = GPT2LMHeadModel.from_pretrained('question_identifier_model')

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fine_tuned_model.to(device)

# Test the model on a prompt
prompt = "Have you ever had surgery?"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
output = fine_tuned_model.generate(input_ids, max_length=50, do_sample=True, temperature=0.7)

# Decode the output
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)