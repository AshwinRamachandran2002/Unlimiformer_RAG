from transformers import AutoTokenizer
import numpy as np
import json

model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

file_path = "75_kv.txt"

# define the partitioning token that seperates each fundamental unit of the data
partitioner = ","
# define what the prefix to the structure must be
prefix = "JSON data: "
# start-index in file for data (from start)
start_index = 1
# end-index in file for data (from end)
end_index = 1

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()
    text = text[start_index:-end_index]

tokens = tokenizer.tokenize(text)
token_count = len(tokens)
print(f"Total tokens in the file: {token_count}")

# def split_list_by_value(input_list: list[str], partition: str):
#     input_arr = np.array(input_list)
#     indices = np.where(input_arr == partition)[0]
#     subarrays = np.split(input_arr, indices + 1)
#     sublists = [sub.tolist() for sub in subarrays]
#     return sublists
# split_segments = split_list_by_value(tokens, partitioner_token[0])

segments = text.split(partitioner)
tokenized_segments = [tokenizer.encode(segment.strip()) for segment in segments]
token_count_per_segment = [len(segment) for segment in tokenized_segments]
# print(token_count_per_segment)

config = {
    "segment_length": token_count_per_segment
}
with open("config_data_" + file_path  + ".json", "w") as f:
    json.dump(config, f)