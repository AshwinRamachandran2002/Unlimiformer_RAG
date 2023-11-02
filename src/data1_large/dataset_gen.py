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
                fact = f"Fact number {numbers[i]}: {persons[i]} is {actions[i]}"
                output_file.write(fact)
            else:
                fact = f"Fact number {numbers[i]}: {persons[i]} is {actions[i]},"
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
for fact in facts:
    tokens = tokenizer.tokenize(fact)
    token_count = len(tokens)
    token_counts.append(token_count + 1)

# Save the token counts as a JSON file
with open("token_counts.json", "w") as json_file:
    json.dump({"fact_lengths": token_counts}, json_file, indent=4)

print("Token counts have been saved in token_counts.json.")