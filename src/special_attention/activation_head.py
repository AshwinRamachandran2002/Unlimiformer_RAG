import subprocess
import sys

if len(sys.argv) != 3:
    print("Please provide the layer number as a command-line argument.")
    sys.exit(1)

layer_number = sys.argv[1]
input_file = sys.argv[2]
# input_file = "trial1/ext_20_layer_NONE_full.txt"
output_file = "filtered_lines.txt"

grep_command = f'grep -n -A 4 -e "layer {layer_number}$" -e "\[\'" "{input_file}" > "{output_file}"'
subprocess.run(grep_command, shell=True)

with open("filtered_lines.txt", "r") as f:
    s = f.read()
lines = s.split('\n')

last_pre = ""
for i in range(0, len(lines), 6):
    seg = lines[i:i+5]
    if '[' not in seg[0]:
        head = seg[1].split(' ')[-1]
        numbers = seg[2].split('tensor')[1].split('device')[0][2:-3].split(', ')
        numbers = [float(number) for number in numbers]
        tokens = seg[4].split('Unlimiformer')[-1]
        if numbers[0] >= 0.1:
            print("Head", head, "scored", numbers)
            print("Last token", last_pre)
            print("tokens", tokens)
            print()
    else:
        tokens = seg[0].split('Unlimiformer')[-1][3:]
        probs = seg[1].split('tensor')[1].split('device')[0][4:-1].split(', ')
        numbers = [float(number) for number in probs]
        last_pre = tokens
    