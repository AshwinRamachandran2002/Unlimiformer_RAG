#!/bin/bash
list_of_lists=("1")
cuda_list=("3")
for index in "${!list_of_lists[@]}"; do
    list="${list_of_lists[$index]}"
    cuda="${cuda_list[$index]}"
    file_name="better/logs_20uuids_anchor_2_temp_3_$list.txt"
    echo "$file_name"
    python run_generation.py --model_type llama --model_name_or_path meta-llama/Llama-2-13b-chat-hf --prefix "" --tokens_ind "$list" --prompt data_final_data2/original_data.txt --suffix "" --test_unlimiformer --fp16 --length 300 --layer_begin 0 --index_devices "$cuda" --datastore_device "$cuda" 2> "$file_name" &
done