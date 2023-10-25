import sys
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from transformers import AutoTokenizer
 
# total arguments
log = sys.argv[1]
layer = int(sys.argv[2])
head = int(sys.argv[3])


model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
    
attn_wts = torch.load('attn_wts_' + log + '.pt', map_location=lambda storage, loc: storage.cuda(4))
inputs = torch.load('inputs_' + log + '.pt', map_location=lambda storage, loc: storage.cuda(4))

print(len(attn_wts), attn_wts[0].shape, attn_wts[-1].shape)
print(len(inputs), inputs[0])

# In total of 40 layers in Llama-2
# Dividing list of attention weights to per token tensor 
# [(40, x) ..] -> [(num_layers, 40, x) ..]
num_layers = 40
per_phrase_weight_list = []
for phrase_index in range(0, len(attn_wts), num_layers):
    per_phrase_weight_list.append(torch.stack(attn_wts[phrase_index : phrase_index + num_layers]))

layers_to_analyze = [layer]
heads_to_analyze = [head]
heatmap_list = []
for phrase in range(len(per_phrase_weight_list)):
    # print("for token", tokenizer.decode(inputs[attn_wts[0].shape[1] + token - 1])) 
    attn_scores_per_phrase = per_phrase_weight_list[phrase]
    for layer in layers_to_analyze:
        attn_scores_per_phrase_per_layer = attn_scores_per_phrase[layer]
        for head in heads_to_analyze:
            attn_scores_per_phrase_per_layer_per_head = attn_scores_per_phrase_per_layer[head]
            heatmap_list.append(attn_scores_per_phrase_per_layer_per_head)

for phrase in range(len(per_phrase_weight_list)):
    phrase_length = len(heatmap_list[phrase])
    plt.imshow(heatmap_list[phrase].detach().to('cpu').numpy(), cmap='YlGnBu', interpolation='nearest')
    plt.colorbar()

    plt.xticks(range(phrase_length), [f'{i}' for i in range(phrase_length)], fontsize=5)
    # plt.yticks(range(phrase_length), [f'{i}' for i in range(phrase_length)])

    plt.xlabel('X-Axis Label')
    plt.ylabel('Y-Axis Label')

    plt.title('Heatmap from Score Tensor')
    plt.show()
    print('ll')
    plt.savefig('113_' + str(phrase))
    plt.clf()


# for phrase in [1]:
#     phrase_length = len(heatmap_list[phrase])
#     print(torch.topk(heatmap_list[1][25:][:25], 10).indices)