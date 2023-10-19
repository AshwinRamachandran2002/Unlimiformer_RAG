import argparse
import uuid
import random
from transformers import AutoTokenizer
import json

# Set a random seed for reproducibility
random.seed(42)
random.seed(42)

# Step 1: Generate UUID-based key-value pairs
def generate_uuid_pairs(num_pairs):
    # data = {"a": "b", "c": "d"}
    # data = "CREATE TABLE stadium (stadium_id number, location text, name text, capacity number); CREATE TABLE singer (singer_id number, name text, country text, song_name text, song_release_year text, age number, is_male others); "
    # data = "Williamson is baking, Oppenheimer is cycling, Leechenbaum is painting, Zelensky is relaxing"
    # str_data = [for _ in range(num_pairs)]
    data = {str(uuid.uuid4())[:10]:str(uuid.uuid4())[:10] for _ in range(num_pairs)}
    str_data = []
    for key in data:
        str_data.append('{"' + str(key)+ '" : "' + str(data[key]) + '"}' )
    return data, str_data

# Step 1.5: Generate pair lengths
def segment_len(text):
    # model_name = "NumbersStation/nsql-llama-2-7B"
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    partitioner = ", "
    # segments = text.split(partitioner)[:-1]
    segments = text.split(partitioner)
    tokenized_segments = [tokenizer.encode(segment.strip()) for segment in segments]
    token_count_per_segment = [len(segment) for segment in tokenized_segments]
    config = {
        "segment_length": token_count_per_segment
    }
    with open("data9/config_data.json", "w") as f:
        json.dump(config, f)

# Step 2: Randomly sample keys and values
def sample_data(data, num_samples):
    keys = random.sample(list(data.keys()), num_samples)
    values = [data[key] for key in keys]
    return keys, values

# Step 3: Save data to files
def save_data_to_files(data, keys, values, str_data):
    # Save the original data as a Python dictionary in a txt file
    with open('data9/original_data.txt', 'w') as file:
        file.write(', '.join(str_data))

    # Save the sampled keys to a newline-separated txt file
    with open('data9/sampled_keys.txt', 'w') as file:
        file.write('\n'.join(keys))

    # Save the sampled values to a newline-separated txt file
    with open('data9/sampled_values.txt', 'w') as file:
        file.write('\n'.join(values))

def save_data_to_files_only(data):
    # Save the original data as a Python dictionary in a txt file
    with open('data9/original_data.txt', 'w') as file:
        json.dump(data, file)

# Step 4: Import and call the 'main' function
def call_main_function():
    from run_generation import main
    result = main()
    return result

# Step 5: Save the results to a "predictions.txt" file
def save_predictions(result):
    with open('data9/predictions.txt', 'w') as file:
        file.write(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Generation and Processing')
    parser.add_argument('--num_pairs', type=int, help='Number of key-value pairs', required=True)
    parser.add_argument('--num_samples', type=int, help='Number of samples to extract', required=True)
    args = parser.parse_args()

    data, str_data = generate_uuid_pairs(args.num_pairs)
    segment_len(', '.join(str_data))
    keys, values = sample_data(data, args.num_samples)
    save_data_to_files(data, keys, values, str_data)
    # save_data_to_files_only(data)

    # result = call_main_function()
    # save_predictions(result)
