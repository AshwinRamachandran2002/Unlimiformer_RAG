#!/bin/bash
# python3 activation_head.py $1 trial1/ext_20_layer_extra_q4_spaced.txt > extra.txt 
# python3 activation_head.py trial1/step3_2.txt 210 230 > all0.txt

input_file="surfing_2pad_covered.txt"

start_pattern="Below is a list of facts in the format (Fact: <PERSON> is <ACTION>)."

# Initialize variables
counter=0
output_file=""

# Read the input file line by line
while IFS= read -r line; do
    if [[ $line == "$start_pattern" ]]; then
        ((counter++))
        output_file="documents/document_${counter}.txt"
        continue
    fi

    # Write line to the output file (if valid output file is set)
    if [ -n "$output_file" ]; then
        echo "$line" >> "$output_file"
    fi
done < "$input_file"


# python3 activation_head.py trial1/step7.txt 145 160 > all7.txt
# python3 activation_head.py trial1/step8.txt 145 160 > all8.txt
# python3 activation_head.py trial1/step9.txt 45 55 > all9.txt
# python3 activation_head.py trial1/step10.txt 45 55 > all10.txt
# python3 activation_head.py trial1/step11.txt 275 290 > all11.txt
# python3 activation_head.py trial1/step12.txt 275 290 > all12.txt
# python3 activation_head.py trial1/step13.txt 210 225 > all13.txt
# python3 activation_head.py trial1/step5.txt 210 225 > all5.txt



# python3 activation_head.py $1 trial1/step1_zeros.txt > all_zeros.txt
# python3 activation_head.py $1 trial1/ext_20_layer_all_q4_spaced.txt > all.txt
# python3 activation_head.py $1 trial1/ext_20_layer_all_q4_spaced_zeros.txt > all1.txt 
# python3 activation_head.py $1 trial1/ext_20_layer_all_q4_spaced_counteract.txt > all2.txt 
# python3 activation_head.py $1 trial1/ext_20_layer_all_q4_spaced_zeros_dat.txt > all3.txt 
# python3 activation_head.py $1 trial1/ext_20_layer_extra4_q4_spaced_zeros_dat.txt > all4.txt 