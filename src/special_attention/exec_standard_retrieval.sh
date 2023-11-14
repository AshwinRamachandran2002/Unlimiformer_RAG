#!/bin/bash
# cuda_list=("0" "1" "4" "4" "5" "5" "6" "7")
# split_data_list=("False" "False" "True" "True" "True" "True" "True" "True")
# num_extract_list=("20" "30" "20" "30" "20" "30" "20" "30")
# anchor_list=("2" "2" "1" "1" "2" "2" "3" "3")
# template_list=("2" "2" "1" "1" "2" "2" "3" "3")
# data_file_list=("combined_facts.txt" "combined_facts.txt" "combined_facts.txt" "combined_facts.txt" "combined_facts.txt" "combined_facts.txt" "combined_facts.txt" "combined_facts.txt")
# count_file_list=("token_counts.json" "token_counts.json" "token_counts.json" "token_counts.json" "token_counts.json" "token_counts.json" "token_counts.json" "token_counts.json")
cuda_list=("3" "4")
split_data_list=("False" "False")
num_extract_list=("20" "30")
anchor_list=("2" "2")
template_list=("2" "2")
data_file_list=("combined_facts.txt" "combined_facts.txt")
layer_begin_list=("0" "0")
count_file_list=("token_counts.json" "token_counts.json")
for index in "${!cuda_list[@]}"; do
    cuda="${cuda_list[$index]}"
    anchor="${anchor_list[$index]}"
    template="${template_list[$index]}"
    data_file="${data_file_list[$index]}"
    count_file="${count_file_list[$index]}"
    split_data="${split_data_list[$index]}"
    num_extract="${num_extract_list[$index]}"
    layer_begin="${layer_begin_list[$index]}"
    file_name="trial1/ext_${num_extract}_layer_1.txt"
    echo "$file_name"
    python run_generation_retrieval_standard.py --split_data $split_data --num_extract $num_extract --num_anchors $anchor --num_templates $template --data_file "$data_file" --token_count_file "$count_file" --model_type llama --model_name_or_path meta-llama/Llama-2-13b-chat-hf --prefix "" --prompt $data_file --suffix "" --test_unlimiformer --fp16 --length 300 --layer_begin $layer_begin --index_devices "$cuda" --datastore_device "$cuda" 2> "$file_name" &
done