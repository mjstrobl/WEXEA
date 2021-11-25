import json
from unidecode import unidecode

config = json.load(open('../../config/config.json'))
outputpath = config['outputpath']

wexea_directory = outputpath


person_candidates = {}
persons = []
with open('../../data/persons.txt') as f:
    for line in f:
        name = line.strip()
        if "Category" not in name:
            persons.append(name)
            parts = name.split()
            surname = ''

            for part in parts:
                if len(part) > 2 and part[0] != '(':
                    surname = part

            if len(surname) > 0:
                if surname not in person_candidates:
                    person_candidates[surname] = {"dict":{},"list":[]}
                person_candidates[surname]['list'].append((name,-2))
                person_candidates[surname]['dict'][name]= -2

with open(wexea_directory + 'dictionaries/person_candidates.json','w') as f:
    json.dump(person_candidates,f)


with open(wexea_directory + 'dictionaries/persons.json','w') as f:
    json.dump(persons,f)

print("processed person candidates")

aliases = json.load(open(wexea_directory + 'dictionaries/aliases.json'))

new_aliases = {}

for alias in aliases:
    alias_new = unidecode(alias)
    if alias_new not in new_aliases:
        new_aliases[alias_new] = {}

    entities = aliases[alias]
    for entity in entities:
        if entity not in new_aliases[alias_new]:
            new_aliases[alias_new][entity] = 0
        new_aliases[alias_new][entity] += entities[entity]


aliases = new_aliases

with open(wexea_directory + 'dictionaries/aliases_decoded.json','w') as f:
    json.dump(aliases,f)

print("created aliases decoded")

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

with open(wexea_directory + 'dictionaries/aliases_lower.json','w') as f:
    json.dump(aliases_lower,f)


print("created aliases lower.")

num_cands = [10000000]

for n in num_cands:
    print(str(n) + ' candidates.')
    priors_lower = {}
    for alias in aliases_lower:
        entities = aliases_lower[alias]

        entities_list = []
        for entity in entities:
            entities_list.append((entity,entities[entity]))

        entities_list.sort(key=lambda x: x[1], reverse=True)

        all = 0
        for i in range(min(n,len(entities_list))):
            all += entities_list[i][1]

        priors_lower[alias] = {"dict": {}, "list": []}
        for i in range(min(n,len(entities_list))):
            prior = entities_list[i][1] / all
            priors_lower[alias]['list'].append((entities_list[i][0], prior))
            priors_lower[alias]['dict'][entities_list[i][0]] = prior

        #priors_lower[alias]['list'].sort(key=lambda x: x[1],reverse=True)

    filename = wexea_directory + 'dictionaries/priors_lower.json'
    if n < 100:
        filename = wexea_directory + 'dictionaries/priors_lower_' + str(n) + '.json'

    with open(filename,'w') as f:
        json.dump(priors_lower,f)

    print("created priors lower")

    priors = {}
    for alias in aliases:
        entities = aliases[alias]
        entities_list = []
        for entity in entities:
            entities_list.append((entity, entities[entity]))

        entities_list.sort(key=lambda x: x[1], reverse=True)

        all = 0
        for i in range(min(n, len(entities_list))):
            all += entities_list[i][1]

        priors[alias] = {"dict": {}, "list": []}
        for i in range(min(n, len(entities_list))):
            prior = entities_list[i][1] / all
            priors[alias]['list'].append((entities_list[i][0], prior))
            priors[alias]['dict'][entities_list[i][0]] = prior

        #priors[alias]['list'].sort(key=lambda x: x[1],reverse=True)

    filename = wexea_directory + 'dictionaries/priors.json'
    if n < 100:
        filename = wexea_directory + 'dictionaries/priors_' + str(n) + '.json'

    with open(filename, 'w') as f:
        json.dump(priors, f)

print("created priors")





title2filename = {}
filename2title = json.load(open(wexea_directory + 'dictionaries/filename2title_2.json'))

for filename in filename2title:
    title2filename[filename2title[filename]] = filename

with open(wexea_directory + 'dictionaries/title2filename.json','w') as f:
    json.dump(title2filename,f)





