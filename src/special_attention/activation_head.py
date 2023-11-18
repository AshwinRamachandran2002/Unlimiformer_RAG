import subprocess
import sys

if len(sys.argv) != 4:
    print("Please provide the layer number as a command-line argument.")
    sys.exit(1)

# layer_number = sys.argv[1]
input_file = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
# input_file = "trial1/ext_20_layer_NONE_full.txt"
output_file = "filtered_lines.txt"

selected_heads_per_layer={}
for layer_number in range(4,40):
    
    # grep_command = f'grep -n -E -A 5 "layer {layer_number}$|\[\'" "{input_file}" > "{output_file}"'
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
            head = seg[1].split(' ')[-1]
            numbers = seg[2].split('tensor')[1].split('device')[0][2:-3].split(', ')
            numbers = [float(number) for number in numbers]
            tokens = seg[4].split('Unlimiformer')[-1]
            numbers_tokens = seg[5].split('tensor')[1][2:-2].split(', ')
            numbers_tokens = [float(number) for number in numbers_tokens]

            if numbers_tokens[0] >= start and numbers_tokens[0] <= end and numbers[0] >= 0.1:
                if head not in around_pos:
                    around_pos.append(head)

            if False:#numbers[0] >= 0.1:
                print("Head", head, "scored", numbers)
                print("Last token", last_pre)
                print("tokens", tokens)
                print("tokens number", numbers_tokens)
                print()
                pass
        else:
            try:
                tokens = seg[0].split('Unlimiformer')[-1][3:]
                probs = seg[1].split('tensor')[1].split('device')[0][4:-1].split(', ')
                numbers = [float(number) for number in probs]
                last_pre = tokens
            except:
                pass

    selected_heads_per_layer[layer_number] = around_pos
print("Around Positions Head", selected_heads_per_layer)

flags_per_layer = {}
for layer_number in range(4,40):
    
    # grep_command = f'grep -n -E -A 5 "layer {layer_number}$|\[\'" "{input_file}" > "{output_file}"'
    grep_command = f'grep -n -A 5 -e "layer {layer_number}$" -e "\[\'" "{input_file}" > "{output_file}"'
    subprocess.run(grep_command, shell=True)

    with open("filtered_lines.txt", "r") as f:
        s = f.read()
    lines = s.split('\n')

    last_pre = ""
    
    flags = []
    for i in range(0, len(lines), 7):
        seg = lines[i:i+6]
        if '[' not in seg[0]:
            head = seg[1].split(' ')[-1]
            numbers = seg[2].split('tensor')[1].split('device')[0][2:-3].split(', ')
            numbers = [float(number) for number in numbers]
            tokens = seg[4].split('Unlimiformer')[-1]
            numbers_tokens = seg[5].split('tensor')[1][2:-2].split(', ')
            numbers_tokens = [float(number) for number in numbers_tokens]
    
            if not(numbers_tokens[0] >= 0 and numbers_tokens[0] <= 10):
            # if numbers_tokens[0] >= start and numbers_tokens[0] <= end:
                pass
            elif numbers[0] >= 0.1 and head in selected_heads_per_layer[layer_number]:
                print("FLAG!!")
                print("layer", layer_number)
                print("head", head, "predicting", numbers_tokens[0], "score", numbers[0])
                if head not in flags:
                    flags.append(head)
                # flags.append((head, tokens))
                print("Head", head, "scored", numbers)
                print("Last token", last_pre)
                print("tokens", tokens)
                print("tokens number", numbers_tokens)
                print()
                pass
        else:
            try:
                tokens = seg[0].split('Unlimiformer')[-1][3:]
                probs = seg[1].split('tensor')[1].split('device')[0][4:-1].split(', ')
                numbers = [float(number) for number in probs]
                last_pre = tokens
            except:
                pass

    flags_per_layer[layer_number] = flags

print("Flags Heads", flags_per_layer)

final_per_layer = {}
for layer_number in range(4,40):
    final_per_layer[layer_number] = list(set(selected_heads_per_layer[layer_number]) - set(flags_per_layer[layer_number]))

print(final_per_layer)
    # for i in range(0, len(lines), 7):
    #     seg = lines[i:i+6]
    #     # print(seg)
    #     if '[' not in seg[0]:
    #         # try:
    #             head = seg[1].split(' ')[-1]
    #             numbers = seg[2].split('tensor')[1].split('device')[0][2:-3].split(', ')
    #             numbers = [float(number) for number in numbers]
    #             tokens = seg[4].split('Unlimiformer')[-1]
    #             numbers_tokens = seg[5].split('tensor')[1][2:-2].split(', ')
    #             # print(seg[5])
    #             numbers_tokens = [float(number) for number in numbers_tokens]
    #             if numbers_tokens[0] >= 210 and numbers_tokens[0] <= 230:
    #                 if head not in around_pos:
    #                     around_pos.append(head)
    #             if numbers[0] >= 0.01 and head in flags:
    #                 print("Head", head, "scored", numbers)
    #                 print("Last token", last_pre)
    #                 print("tokens", tokens)
    #                 print("tokens number", numbers_tokens)
    #                 print()
    #                 pass
    #         # except:
    #         #     pass
    #     else:
    #         try:
    #             tokens = seg[0].split('Unlimiformer')[-1][3:]
    #             probs = seg[1].split('tensor')[1].split('device')[0][4:-1].split(', ')
    #             numbers = [float(number) for number in probs]
    #             last_pre = tokens
    #         except:
    #             pass

    # a[layer_number] = around_pos
    # print(a)