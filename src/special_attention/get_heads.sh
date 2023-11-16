#!/bin/bash

input_file="trial1/ext_20_layer_NONE_full.txt"  # Replace with your file name
output_file="filtered_lines.txt"  # Replace with the desired output file name
extracted_heads="extracted_heads.txt"  # File to store extracted heads

# Extract lines containing "layer 20" and the subsequent 3 lines using grep
grep -n -A 3 "layer 30$" "$input_file" > "$output_file"