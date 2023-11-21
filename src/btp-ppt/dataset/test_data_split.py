import json
import torch
from transformers import AutoTokenizer

with open("token_counts_pad.json", "r") as f:
    text = f.read()
    parsed_data = json.loads(text)
    segment_lengths = parsed_data["fact_lengths"]

model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

prefix = "\n"
prefix_input_id = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")

results = []
context_start = 0
context_end = segment_lengths[0] + 1
results.append((context_start, context_end - 1, -segment_lengths[0], None ))  

for i in range(1, len(segment_lengths)):
    context_start = context_start + segment_lengths[i - 1]
    context_end = context_end + segment_lengths[i]
    
    update_start_ind = -segment_lengths[i]
    update_end_ind = None
    
    cs, ce, us, ue = context_start, context_end  - 1, update_start_ind, update_end_ind
    results.append((cs, ce, us, ue))


with open("combined_facts_pad.txt", "r") as f:
    text = f.read()
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    print(input_ids)

for context_start_ind, context_end_ind, update_start_ind, update_end_ind in results:
    print(f'Encoding {context_start_ind} to {context_end_ind} out of {input_ids.shape[-1]}')
    chunk = input_ids[:, context_start_ind:context_end_ind]
    print(tokenizer.decode(chunk[0]))
    chunk = chunk[:, update_start_ind:update_end_ind]
    print(f'Encoding {update_start_ind} to {update_end_ind} out of {input_ids.shape[-1]}')
    print(tokenizer.decode(chunk[0]))
    print(len(chunk[0]))