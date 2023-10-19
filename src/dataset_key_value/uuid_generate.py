import uuid
import argparse
import json

parser = argparse.ArgumentParser(description ='Process some integers.')
parser.add_argument('number', type = int)
args = parser.parse_args()  

dict_kv = {}
for _ in range(args.number):
    dict_kv[str(uuid.uuid4())] = str(uuid.uuid4())

with open(str(args.number) + "_kv.txt", 'w+') as f:
    json.dump(dict_kv, f)