# %%
import gzip
from collections import defaultdict
import json

# %%
dataset = []

for l in gzip.open("train.json.gz", 'rt', encoding='utf-8'):
    d = eval(l)
    dataset.append(d)

print(dataset[0])


