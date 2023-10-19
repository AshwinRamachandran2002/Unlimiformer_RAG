import json
import torch
from transformers import AutoTokenizer

with open("data6/config_data.json", "r") as f:
# with open("dataset_key_value/config_data_75_kv.txt.json", "r") as f:
    text = f.read()
    parsed_data = json.loads(text)
    segment_lengths = parsed_data["segment_length"]
# model_name = "NumbersStation/nsql-llama-2-7B"
model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prefix = "JSON Data: {"
prefix = ""
prefix_input_id = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")
len_prefix_id = len(prefix_input_id[0])

results = []
context_start = 0
context_end = segment_lengths[0] - 2
# context_end = segment_lengths[0]
results.append((context_start, context_end + 1, -segment_lengths[0] + 1, None ))  
# results.append((context_start, context_end - 1, len_prefix_id, len_prefix_id + segment_lengths[0] - 1 ))  

for i in range(1, len(segment_lengths)):
    context_start = context_start + (segment_lengths[i - 2] if i > 1 else 0)
    context_end = context_end + segment_lengths[i]
    
    update_start_ind = -segment_lengths[i] + 1
    update_end_ind = None
    
    # cs, ce, us, ue = context_start - (0 if i > 1 else 0), context_end, update_start_ind, update_end_ind
    cs, ce, us, ue = context_start - (1 if i > 1 else 0), context_end + 1, update_start_ind - 1, update_end_ind
    # cs, ce, us, ue = context_start - 1, context_end - 1, update_start_ind + len_prefix_id, update_end_ind + len_prefix_id - 1
    # cs, ce, us, ue = context_start, context_end - 1, update_start_ind + len_prefix_id, update_end_ind + len_prefix_id - 1
    results.append((cs, ce, us, ue))


with open("data6/original_data.txt", "r") as f:
# with open("dataset_key_value/75_kv.txt", "r") as f:
    text = f.read()
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    print(input_ids)
    # print(input_ids.attention_mask)

for context_start_ind, context_end_ind, update_start_ind, update_end_ind in results:
    # print(f'Encoding {context_start_ind} to {context_end_ind} out of {input_ids.shape[-1]}')
    chunk = input_ids[:, context_start_ind:context_end_ind]
    # chunk = torch.cat((prefix_input_id, input_ids[:, context_start_ind:context_end_ind]), dim = 1)
    print(tokenizer.decode(chunk[0]))
    print(len(chunk[0]))
    chunk = chunk[:, update_start_ind:update_end_ind]
    # print(f'Encoding {update_start_ind} to {update_end_ind} out of {input_ids.shape[-1]}')
    # chunk = chunk[:, update_start_ind + 3:update_end_ind]
    print(tokenizer.decode(chunk[0]))
    print(len(chunk[0]))