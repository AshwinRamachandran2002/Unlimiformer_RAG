import subprocess
import sys

# if len(sys.argv) != 2:
#     print("Please provide the layer number as a command-line argument.")
#     sys.exit(1)

layer_wise = {}
for layer_number in range(40):
    # layer_number = sys.argv[1]
    print(f"Layer: {layer_number}")
    input_file = "trial1/ext_20_layer_NONE_full.txt"
    output_file = "filtered_lines.txt"
    extracted_heads = "extracted_heads.txt"

    grep_command = f'grep -n -A 3 "layer {layer_number}$" "{input_file}" > "{output_file}"'
    subprocess.run(grep_command, shell=True)

    with open("filtered_lines.txt", "r") as f:
        s = f.read()
    lines = s.split('\n')
    heads_count = {}
    for i in range(0, len(lines), 5):
    # for i in range(int(len(lines) * (3/4))//5 * 5, int(len(lines) * (4/4))//5 * 5, 5):
        seg = lines[i:i+4]
        head = seg[1].split(' ')[-1]
        numbers = seg[2].split('tensor')[1].split('device')[0][2:-3].split(', ')
        numbers = [float(number) for number in numbers]
        if numbers[0] >= 0.01:
            if head in heads_count:
                heads_count[head] += 1
            else:
                heads_count[head] = 1
    sorted_heads_count = dict(sorted(heads_count.items(), key=lambda x: x[1], reverse=True))
    for head, count in sorted_heads_count.items():
        print(f"Head: {head}, Count: {count}")
    layer_wise[layer_number] = list(map(int, sorted_heads_count.keys()))
print(layer_wise)