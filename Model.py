import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the generation parameters
max_length = 30
num_return_sequences = 3
temperature = 0.6  # Increase temperature to generate more diverse responses
top_p = 0.9  # Introduce top-p sampling to limit the set of tokens considered at each step
top_k = 50  # Further limit the set of tokens considered by setting a maximum number to sample from

# Encode the input question
prompt = "Is the following a yes/no question? Would you like coffee?"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate the responses
output_ids = model.generate(
    input_ids=input_ids,
    max_length=max_length,
    num_return_sequences=num_return_sequences,
    temperature=temperature,
    no_repeat_ngram_size=2,
    repetition_penalty=1.0,
    do_sample=True,
    top_p=top_p,
    top_k=top_k
)

# Decode and print the responses
for i, output in enumerate(output_ids):
    print(f"{i+1}. {tokenizer.decode(output, skip_special_tokens=True)}")

