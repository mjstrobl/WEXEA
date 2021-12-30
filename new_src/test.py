import json
from utils import is_upper


config = json.load(open('../config/config.json'))
outputpath = config['outputpath']
dictionarypath = outputpath + 'dictionaries/'


aliases = json.load(open(dictionarypath + 'aliases.json'))

priors = {}
for alias in aliases:
    candidates = aliases[alias]
    all = 0
    for candidate in candidates:
        all += candidates[candidate]

    l = []
    for candidate in candidates:
        p = candidates[candidate] / all
        l.append((candidate,p))

    l.sort(key=lambda x: x[1], reverse=True)

    priors[alias] = l

with open(dictionarypath + 'priors.json','w') as f:
    json.dump(priors,f)


