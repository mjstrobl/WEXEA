import json


config = json.load(open('../../config/config.json'))
outputpath = config['outputpath']

wexea_directory = outputpath


aliases = json.load(open(wexea_directory + 'dictionaries/aliases.json'))

aliases_lower = {}

for alias in aliases:
    entities = aliases[alias]
    alias = alias.lower()
    for entity in entities:
        if entity.startswith("Category:"):
            continue
        c = entities[entity]
        if alias not in aliases_lower:
            aliases_lower[alias] = {}
        if entity not in aliases_lower[alias]:
            aliases_lower[alias][entity] = 0
        aliases_lower[alias][entity] += c

del aliases

print("created aliases lower.")

priors_lower = {}
for alias in aliases_lower:
    entities = aliases_lower[alias]
    all = 0
    for entity in entities:
        all += entities[entity]

    priors_lower[alias] = []
    for entity in entities:
        prior = entities[entity] / all
        priors_lower[alias].append((entity,prior))

    priors_lower[alias].sort(key=lambda x: x[1],reverse=True)

with open(wexea_directory + 'dictionaries/priors_lower.json','w') as f:
    json.dump(priors_lower,f)

person_candidates = {}

with open('../../data/persons.txt') as f:
    for line in f:
        name = line.strip()
        if "Category" not in name:
            parts = name.split()
            for part in parts:
                if len(part) > 2 and part[0] != '(':
                    if part.lower() not in person_candidates:
                        person_candidates[part.lower()] = []
                    person_candidates[part.lower()].append(name)

with open(wexea_directory + 'dictionaries/person_candidates.json','w') as f:
    json.dump(person_candidates,f)

print("processed person candidates")



title2filename = {}
filename2title = json.load(open(wexea_directory + 'dictionaries/filename2title_2.json'))

for filename in filename2title:
    title2filename[filename2title[filename]] = filename

with open(wexea_directory + 'dictionaries/title2filename.json','w') as f:
    json.dump(title2filename,f)





