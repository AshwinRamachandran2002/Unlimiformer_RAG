import random

with open('persons.txt', 'r') as persons_file:
    persons = persons_file.read().splitlines()

with open('actions.txt', 'r') as actions_file:
    actions = actions_file.read().splitlines()

with open('extra_locations.txt', 'r') as extra_locations_file:
    extra_locations = extra_locations_file.read().splitlines()

with open('extra_persons.txt', 'r') as extra_persons_file:
    extra_persons = extra_persons_file.read().splitlines()


pad_persons = persons[:7]
for i in range(7, len(persons)-7):
    pad_persons.append(persons[i])
    for j in range(0):
        pad_person = random.choice(extra_persons)
        while pad_person in pad_persons:
            pad_person = random.choice(extra_persons)
        pad_persons.append(pad_person)
pad_persons.extend(persons[-7:])


pad_actions = actions[:7]
for i in range(7, len(actions)-7):
    pad_actions.append(actions[i])
    for j in range(0):
        pad_location = random.choice(extra_locations)
        while pad_location in pad_actions:
            pad_location = random.choice(extra_locations)
        pad_actions.append(pad_location)
pad_actions.extend(actions[-7:])


if len(persons) != len(actions):
    print("Error: The input files have different numbers of lines.")
else:
    # Create a new text file to write the combined content
    with open('combined_facts_pad.txt', 'w') as output_file:
        for i in range(len(pad_persons)):
            if i == len(pad_persons) - 1:
                fact = f"Fact: {pad_persons[i]} is {pad_actions[i]}"
                output_file.write(fact)
            elif i == 0:
                fact = f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Answer with short responses according to the question. \n<</SYS>>\n\nBelow is a list of facts in the format (Fact: <PERSON> is <ACTION>).\nFact: {persons[i]} is {actions[i]}, "
                output_file.write(fact)
            else:
                if pad_actions[i] in extra_locations:
                    fact = f"Fact: {pad_persons[i]} is in {pad_actions[i]},"
                else:
                    fact = f"Fact: {pad_persons[i]} is {pad_actions[i]},"
                output_file.write(fact + ' ')

print("Facts have been combined and saved in combined_facts_pad.txt.")


############################################################################

from transformers import AutoTokenizer
import json

# Load a pre-trained tokenizer, e.g., BERT
model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Read the combined_facts.txt file
with open("combined_facts_pad.txt", "r") as file:
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
with open("token_counts_pad.json", "w") as json_file:
    json.dump({"fact_lengths": token_counts}, json_file, indent=4)

print("Token counts have been saved in token_counts_pad.json.")