import subprocess
import sys
import json
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("../dataset/token_counts_pad.json", "r") as f:
    leng = json.loads(f.read())["fact_lengths"]

with open("../dataset/combined_facts_pad.txt", "r") as f:
    text = f.read()
    text = [tokenizer.decode(x) for x in tokenizer.encode(text)][leng[0]:]

leng=leng[1:]
output_file = "filtered_lines.txt"

final_flags_per_layer = {}
final_selected_heads_per_layer = {}
for k in range(7 + 3*16, 7 + 3*16 + 2 , 2):
    for j in [2, 3]:

        input_file = f"documents/document_{j}.txt"
        
        start = sum(leng[:k-1]) - 5
        end = sum(leng[:k]) + 5
        print("Fact Positions", start, "to", end)

        selected_heads_per_layer={}
        for layer_number in range(1,40):
            
            grep_command = f'grep -n -A 5 -e "layer {layer_number}$" -e "\[\'" "{input_file}" > "{output_file}"'
            subprocess.run(grep_command, shell=True)

            with open("filtered_lines.txt", "r") as f:
                s = f.read()
            lines = s.split('\n')

            last_pre = ""

            around_pos = []
            for i in range(0, len(lines), 7):
                seg = lines[i:i+6]

                if '[' not in seg[0]:
                    head = int(seg[1].split(' ')[-1])
                
                    numbers = seg[2].split('tensor')[1].split('device')[0][2:-3].split(', ')
                    numbers = [float(number) for number in numbers]
                
                    tokens = seg[4].split('Unlimiformer')[-1]
                
                    numbers_tokens = seg[5].split('tensor')[1][2:-2].split(', ')
                    numbers_tokens = [int(number) for number in numbers_tokens]

                    if numbers_tokens[0] >= start and numbers_tokens[0] <= end and numbers[0] >= 0.1:
                        if head not in around_pos:
                            around_pos.append(head)

                    if False:#numbers[0] >= 0.1:
                        print("Head", head, "scored", numbers)
                        print("Last token", last_pre)
                        print("tokens", tokens)
                        print("tokens number", numbers_tokens)
                        print()
                else:
                    tokens = seg[0].split('Unlimiformer')[-1][3:]
                    probs = seg[1].split('tensor')[1].split('device')[0][4:-1].split(', ')
                    numbers = [float(number) for number in probs]
                    last_pre = tokens
            
            around_pos.sort()
            selected_heads_per_layer[layer_number] = around_pos

        # print("Around Positions Head", selected_heads_per_layer)

        flags_per_layer = {}
        for layer_number in range(1,40):
            
            grep_command = f'grep -n -A 5 -e "layer {layer_number}$" -e "\[\'" "{input_file}" > "{output_file}"'
            subprocess.run(grep_command, shell=True)

            with open("filtered_lines.txt", "r") as f:
                s = f.read()
            lines = s.split('\n')

            last_pre = ""
            
            flags = {}
            seq = 0
            for i in range(0, len(lines), 7):
                seg = lines[i:i+6]
                if '[' not in seg[0]:
                    head = int(seg[1].split(' ')[-1])
                    numbers = seg[2].split('tensor')[1].split('device')[0][2:-3].split(', ')
                    numbers = [float(number) for number in numbers]
                    tokens = seg[4].split('Unlimiformer')[-1]
                    numbers_tokens = seg[5].split('tensor')[1][2:-2].split(', ')
                    numbers_tokens = [int(number) for number in numbers_tokens]
            
                    if numbers_tokens[0] >= start and numbers_tokens[0] <= end and numbers[0] >=0.1:
                        if head not in flags:
                            flags[head] = [(j, seq, numbers_tokens[0], text[numbers_tokens[0] - 1], text[numbers_tokens[0]], text[numbers_tokens[0] + 1], numbers[0])]
                        else:
                            flags[head].append((j, seq, numbers_tokens[0], text[numbers_tokens[0] - 1], text[numbers_tokens[0]], text[numbers_tokens[0] + 1], numbers[0]))
                    elif numbers[0] >= 0.1 and head in selected_heads_per_layer[layer_number]:
                        # print("FLAG!!")
                        # print("layer", layer_number)
                        # print("head", head, "predicting", numbers_tokens[0], "score", numbers[0])
                        if head not in flags:
                            flags[head] = [(j, seq, numbers_tokens[0], text[numbers_tokens[0] - 1], text[numbers_tokens[0]], text[numbers_tokens[0] + 1], numbers[0])]
                        else:
                            flags[head].append((j, seq, numbers_tokens[0], text[numbers_tokens[0] - 1], text[numbers_tokens[0]], text[numbers_tokens[0] + 1], numbers[0]))

                        # print("Head", head, "scored", numbers)
                        # print("Last token", last_pre)
                        # print("tokens", tokens)
                        # print("tokens number", numbers_tokens)
                        # print()
                        pass
                else:
                    try:
                        tokens = seg[0].split('Unlimiformer')[-1][3:]
                        probs = seg[1].split('tensor')[1].split('device')[0][4:-1].split(', ')
                        numbers = [float(number) for number in probs]
                        last_pre = tokens
                        seq += 1
                    except:
                        pass

            flags_per_layer[layer_number] = flags

        # print("Flags Heads", flags_per_layer)

        if len(final_selected_heads_per_layer.keys()) == 0:
            for layer_number in range(1,40):
                final_selected_heads_per_layer[layer_number] = selected_heads_per_layer[layer_number]
        else:
            for layer_number in range(1,40):
                final_selected_heads_per_layer[layer_number] = list( set(final_selected_heads_per_layer[layer_number]) | set(selected_heads_per_layer[layer_number]))

        if len(final_flags_per_layer.keys()) == 0:
            for layer_number in range(1,40):
                final_flags_per_layer[layer_number] = flags_per_layer[layer_number]
        else:
            for layer_number in range(1,40):
                for head in flags_per_layer[layer_number].keys():
                    if head not in final_flags_per_layer[layer_number]:
                        final_flags_per_layer[layer_number][head] = flags_per_layer[layer_number][head]
                    else:
                        final_flags_per_layer[layer_number][head] = list( set(final_flags_per_layer[layer_number][head]) | set(flags_per_layer[layer_number][head]))
    
print(final_flags_per_layer)
print(final_selected_heads_per_layer)