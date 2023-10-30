#!/bin/bash


# list_of_lists=("0 1 2 3" "0 1 2 3 4 5 6")
# list_of_lists=("0 1 2 3 4 5 6 7 8 9 10" "1 2 3 4 5 6 7 8 9 10" "2" "2 3 4 5 6 7 8 9 10" "3 4 5 6 7 8 9 10")
list_of_lists=("0 1 10" "0 1 3 10" "0 1 3 7 10" "0 10" "3 10" "7 10")
# list_of_lists=("0 1 2 3 9 10" "4 5 6" "4 5 6 7 8 9 10" "0 1 2 3 10" "0 1 2 3 4 5 6 7 8 9 10" "10")
cuda_list=("1" "2" "5" "6" "7" "4")  # CUDA index for each list element
# cuda_list=("0" "1")  # CUDA index for each list element
for index in "${!list_of_lists[@]}"; do
    list="${list_of_lists[$index]}"
    cuda="${cuda_list[$index]}"
    # Split the space-separated string into individual tokens
    file_name="logs_kt_2_roman_numerals_anchor=2_$list.txt"
    echo "$file_name"
    python run_generation.py --model_type llama --model_name_or_path meta-llama/Llama-2-13b-chat-hf --prefix "" --tokens_ind "$list" --prompt data_final_data1/original_data.txt --suffix "" --test_unlimiformer --fp16 --length 300 --layer_begin 0 --index_devices "$cuda" --datastore_device "$cuda" 2> "$file_name" &
done