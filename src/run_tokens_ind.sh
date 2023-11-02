#!/bin/bash
# list_of_lists=("0 1 2 3" "0 1 2 3 4 5 6")
# list_of_lists=("0 1 2 3 4 5 6 7 8 9 10" "0 1 2 3 4 5 6 7 8 10" "0 1 2 3 4 5 6 7 9 10")
# list_of_lists=("0 1 2 3 4 5 6 8 9 10 11 12" "0 1 2 3 4 5 6 7 9 10 11 12" "0 1 2 3 4 5 6 7 8 10 11 12" "0 1 2 3 4 5 6 7 8 9 11 12" "0 1 2 3 4 5 6 7 8 9 10 12" "0 1 2 3 4 5 6 7 8 9 10 11")
list_of_lists=("1")
# list_of_lists=("0" "0 1" "3" "10")
# list_of_lists=("0 1 2 3 4 5 6 7 8 9 10 11 12" "1 2 3 4 5 6 7 8 9 10 11 12" "0 2 3 4 5 6 7 8 9 10 11 12" "0 1 3 4 5 6 7 8 9 10 11 12" "0 1 2 4 5 6 7 8 9 10 11 12" "0 1 2 3 5 6 7 8 9 10 11 12" "0 1 2 3 4 6 7 8 9 10 11 12" "0 1 2 3 4 5 7 8 9 10 11 12" "0 1 2 3 4 5 6 8 9 10 11 12" "0 1 2 3 4 5 6 7 9 10 11 12" "0 1 2 3 4 5 6 7 8 10 11 12" "0 1 2 3 4 5 6 7 8 9 11 12" "0 1 2 3 4 5 6 7 8 9 10 12" "0 1 2 3 4 5 6 7 8 9 10 11")
# list_of_lists=("2")
# list_of_lists=("0 1 2 3 9 10" "4 5 6" "4 5 6 7 8 9 10" "0 1 2 3 10" "0 1 2 3 4 5 6 7 8 9 10" "10")
# cuda_list=("3" "4" "5")  # CUDA index for each list element
cuda_list=("3")  # CUDA index for each list element
# cuda_list=("3" "3" "4" "4")  # CUDA index for each list element
# cuda_list=("0" "1" "2" "3" "3" "3" "4" "7" "4" "5" "7" "5" "6" "6")  # CUDA index for each list element
# cuda_list=("0" "1")  # CUDA index for each list element
for index in "${!list_of_lists[@]}"; do
    list="${list_of_lists[$index]}"
    cuda="${cuda_list[$index]}"
    # Split the space-separated string into individual tokens
    file_name="better/logs_20ids_anchor_2_temp_3_$list.txt"
    echo "$file_name"
    python run_generation.py --model_type llama --model_name_or_path meta-llama/Llama-2-13b-chat-hf --prefix "" --tokens_ind "$list" --prompt data_final_data2/original_data.txt --suffix "" --test_unlimiformer --fp16 --length 300 --layer_begin 0 --index_devices "$cuda" --datastore_device "$cuda" 2> "$file_name" &
done