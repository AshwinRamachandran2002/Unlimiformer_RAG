#!/bin/bash
cuda_list=("3")
anchor_list=("4")
template_list=("4")
data_file_list=("combined_facts.txt")
count_file_list=("token_counts.json")
for index in "${!cuda_list[@]}"; do
    cuda="${cuda_list[$index]}"
    anchor="${anchor_list[$index]}"
    template="${template_list[$index]}"
    data_file="${data_file_list[$index]}"
    count_file="${count_file_list[$index]}"
    file_name="final/full_data_file_${data_file}_count_file_${count_file}_retrieval6.txt"
    echo "$file_name"
    python run_generation_retrieval_standard.py --data_file "$data_file" --token_count_file "$count_file" --model_type llama --model_name_or_path meta-llama/Llama-2-13b-chat-hf --prefix "" --prompt $data_file --suffix "" --test_unlimiformer --fp16 --length 300 --layer_begin 0 --index_devices "$cuda" --datastore_device "$cuda" 2> "$file_name" &
done