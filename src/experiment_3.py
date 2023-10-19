import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model_name = "meta-llama/Llama-2-13b-chat-hf"
device = 'cuda:5'
tokenizer = LlamaTokenizer.from_pretrained(model_name)

prefix = "<s>[INST] <<SYS>>\nYou are a helpful assistant. Answer with concise and very very short responses according to the instruction. \n<</SYS>>\n\n"
input_id_prefix = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")
position_id_prefix = torch.arange(0, len(input_id_prefix[0])).unsqueeze(0)

segments = ['{"92a565a1-ba98-44da-bdb6-e91c72cda7fc": "04d6df23-1e04-4d6f-ac99-c88b35bc7041"', '{"bdc28ded-a3e4-4f94-9cdb-b0c42ee45d4d": "413a344f-755e-4239-8cc4-97ca1c685bbb"']
input_id_segments = [tokenizer.encode(segment, add_special_tokens=False, return_tensors="pt") for segment in segments]
position_id_segments = [torch.arange(len(input_id_prefix[0]), len(input_id_prefix[0]) + len(input_id_segment[0])).unsqueeze(0) for input_id_segment in input_id_segments]

val = ''
segment_length_max = 0
for segment in input_id_segments:
    segment_length_max = max(segment_length_max, len(segment[0]))
suffix = 'From the JSON data above, key "' + val + '", value: \n[/INST]'
input_id_suffix = tokenizer.encode(suffix, add_special_tokens=False, return_tensors="pt")
position_id_suffix = torch.arange(len(input_id_prefix[0]) + segment_length_max, len(input_id_prefix[0]) + segment_length_max + len(input_id_suffix[0])).unsqueeze(0)

input_ids = torch.cat((input_id_prefix, torch.cat(input_id_segments, dim = 1), input_id_suffix), dim = 1).to(device)
position_ids = torch.cat((position_id_prefix, torch.cat(position_id_segments, dim = 1), position_id_suffix), dim = 1).to(device)

model = LlamaForCausalLM.from_pretrained(model_name).to(device)

max_length = 300
while input_ids.shape[1] < max_length:
    with torch.no_grad():
        logits = model(input_ids, position_ids=position_ids).logits[:, -1, :]
    next_token_id = torch.argmax(logits, dim=-1)
    input_ids = torch.cat((input_ids, next_token_id.unsqueeze(1)), dim=-1)
    position_ids = torch.cat((position_ids, next_token_id.unsqueeze(1)), dim=-1)
    print(tokenizer.decode(next_token_id))

# Decode the generated sequence
generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(generated_text)