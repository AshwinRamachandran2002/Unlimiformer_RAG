# Read the contents of the three input text files
with open('numbers.txt', 'r') as numbers_file:
    numbers = numbers_file.read().splitlines()

with open('persons.txt', 'r') as persons_file:
    persons = persons_file.read().splitlines()

with open('actions.txt', 'r') as actions_file:
    actions = actions_file.read().splitlines()

# Check that the three lists have the same length
if len(numbers) != len(persons) or len(persons) != len(actions):
    print("Error: The input files have different numbers of lines.")
else:
    # Create a new text file to write the combined content
    with open('combined_facts.txt', 'w') as output_file:
        for i in range(len(numbers)):
            if i == len(numbers) - 1:
                fact = f"Fact: {persons[i]} is {actions[i]}"
                # fact = f"Fact number {numbers[i]}: {persons[i]} is {actions[i]}"
                output_file.write(fact)
            elif i == 0:
                fact = f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Answer with short responses according to the question. \n<</SYS>>\n\nBelow is a list of facts in the format (Fact: <PERSON> is <ACTION>).\nFact: {persons[i]} is {actions[i]}, "
                # fact = f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Answer with short responses according to the question. \n<</SYS>>\n\nBelow is a numbered list of facts in the format (Fact number <FACT-NUMBER>: <PERSON> is <ACTION>).\nFact number {numbers[i]}: {persons[i]} is {actions[i]}, "
                output_file.write(fact)
            else:
                fact = f"Fact: {persons[i]} is {actions[i]},"
                # fact = f"Fact number {numbers[i]}: {persons[i]} is {actions[i]},"
                output_file.write(fact + ' ')

print("Facts have been combined and saved in combined_facts.txt.")


############################################################################

from transformers import AutoTokenizer
import json

# Load a pre-trained tokenizer, e.g., BERT
model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Read the combined_facts.txt file
with open("combined_facts.txt", "r") as file:
    facts = file.read().split(',')

# Initialize a list to store token counts for each statement
token_counts = []

# Process each statement and count tokens
i = 0
for fact in facts:
    tokens = tokenizer.tokenize(fact)
    token_count = len(tokens)
    if i == 0:
        token_counts.append(token_count + 1)
    elif i == len(facts) - 1:
        token_counts.append(token_count - 1)
    else:
        token_counts.append(token_count)
    i += 1

# Save the token counts as a JSON file
with open("token_counts.json", "w") as json_file:
    json.dump({"fact_lengths": token_counts}, json_file, indent=4)

print("Token counts have been saved in token_counts.json.")


###########################################################################

token_ind_to_token_pos = {}
curr_pos = 0
curr_fact = 0
for token_ind in range(sum(token_counts)):
    if curr_pos == token_counts[curr_fact]:
        curr_pos = 0
        curr_fact += 1
    token_ind_to_token_pos[token_ind] = curr_pos
    curr_pos += 1

with open("ind_to_pos.json", "w") as json_file:
    json.dump(token_ind_to_token_pos, json_file, indent=4)

print("Index to Position Mapping has been saved in ind_to_pos.json.")