import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model_name = "meta-llama/Llama-2-13b-chat-hf"
device = 'cuda:5'
tokenizer = LlamaTokenizer.from_pretrained(model_name)

text = "<s>[INST] <<SYS>>\nYou are a helpful assistant. Answer with concise and very very short responses according to the instruction. \n<</SYS>>\n\n how to eat apple? [/INST]"
input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)

model = LlamaForCausalLM.from_pretrained(model_name).to(device)

max_length = 300
while input_ids.shape[1] < max_length:
    with torch.no_grad():
        logits = model(input_ids).logits[:, -1, :]
    next_token_id = torch.argmax(logits, dim=-1)
    input_ids = torch.cat((input_ids, next_token_id.unsqueeze(1)), dim=-1)
    print(tokenizer.decode(next_token_id))

# Decode the generated sequence
generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(generated_text)