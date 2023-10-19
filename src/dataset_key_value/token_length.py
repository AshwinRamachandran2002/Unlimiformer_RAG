from transformers import AutoTokenizer

model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# file_path = "../../example_inputs/harry_potter_full.txt"
file_path = "75_kv.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

tokens = tokenizer.tokenize(text)
token_count = len(tokens)

print(f"Total tokens in the file: {token_count}")