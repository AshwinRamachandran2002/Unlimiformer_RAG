"""
MWE showing that logits from generate match those from forward, except for the first token?
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributions import Categorical
import torch as t

model_name = "meta-llama/Llama-2-13b-chat-hf"

lm = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")


tokenizer.pad_token = tokenizer.eos_token
prompt = tokenizer(["big unpadded five token prompt ", "padded three token "], return_tensors='pt', padding=True, add_special_tokens=True)

#generate with plain sampling (https://huggingface.co/blog/how-to-generate)

result = lm.generate(prompt["input_ids"], attention_mask=prompt["attention_mask"], do_sample=True, output_scores=True, return_dict_in_generate=True, top_k=0, max_length=10)
x, logits_gen = result.sequences, result.scores
logits_gen = t.stack(logits_gen, 1)

x_attention_mask = (x != tokenizer.eos_token_id).to(dtype=t.int64)
position_ids = x_attention_mask.cumsum(-1)-1
position_ids.masked_fill_(x_attention_mask == 0, 1)
print("Attention mask for prompt + generated text")
print(x_attention_mask)
print("Position IDs")
print(position_ids)
logits_for = lm(x, attention_mask=x_attention_mask, position_ids=position_ids).logits
#we drop the last element, and the first prompt_length-1 elements to get
#logits from forward to match those from generate
logits_for = logits_for[:, (prompt["input_ids"].shape[-1]-1):-1]

P_for = Categorical(logits = logits_for)
P_gen = Categorical(logits = logits_gen)

#Take only generated tokens
x = x[..., prompt['input_ids'].shape[-1]:]
log_prob_for = P_for.log_prob(x)
log_prob_gen = P_gen.log_prob(x)

print("log-probs from forward")
print(log_prob_for)
print("log-probs from generate")
print(log_prob_gen)