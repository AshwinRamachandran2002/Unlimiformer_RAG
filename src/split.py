import torch
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

data_file = "data_final_data1"
with open(data_file + "/original_data.txt", 'r') as file:
    file_contents = file.read()

inputs = tokenizer.encode(file_contents)
for i in range(len(inputs)):
    print(tokenizer.decode(inputs[i]), end=" ")
print(len(inputs))

segments = file_contents.split(',')
len_list = []
for segment in segments:
    inputs = tokenizer.encode(segment.strip())
    for i in range(len(inputs)):
        print(tokenizer.decode(inputs[i]), end=" ")
    print(len(inputs))
    len_list.append(len(inputs))
tot_len = 0
len_list[len(len_list)-1] = len_list[len(len_list) - 1] - 1
print(len_list)
for i in range(len(len_list)):
    tot_len += len_list[i]
print(tot_len)
config = {
    "segment_length": len_list
}
import json 
# with open(data_file +"/config_data.json", "w") as f:
#     json.dump(config, f)