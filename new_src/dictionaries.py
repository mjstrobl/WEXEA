import json

dictionary_path = ''

disambiguations_geo = json.load(open(dictionary_path + 'disambiguations_geo.json'))
disambiguations_human = json.load(open(dictionary_path + 'disambiguations_human.json'))
disambiguations_number = json.load(open(dictionary_path + 'disambiguations_number.json'))
disambiguations_other = json.load(open(dictionary_path + 'disambiguations_other.json'))

disambiguations = len(disambiguations_other) + len(disambiguations_number) + len(disambiguations_geo) + len(disambiguations_human)

category2title = json.load(open(dictionary_path + 'category2title.json'))
listof2title = json.load(open(dictionary_path + 'listof2title.json'))
stubs = json.load(open(dictionary_path + 'stubs.json'))
aliases = json.load(open(dictionary_path + 'aliases_pruned.json'))
links = json.load(open(dictionary_path + 'links_pruned.json'))
redirects = json.load(open(dictionary_path + 'redirects_pruned.json'))
given_names = json.load(open(dictionary_path + 'given_names.json'))
surnames = json.load(open(dictionary_path + 'surnames.json'))



print("Surnames: " + str(len(surnames)))
print("given_names: " + str(len(given_names)))
print("redirects: " + str(len(redirects)))
print("aliases: " + str(len(aliases)))
print("stubs: " + str(len(stubs)))
print("links: " + str(len(links)))
print("listof2title: " + str(len(listof2title)))
print("category2title: " + str(len(category2title)))
print("disambiguations: " + str(disambiguations))

categories = json.load(open(dictionary_path + 'categories.json'))

all = 0
for entity in links:
    d = links[entity]
    for e in d:
        all += d[e]

print('all links: ' + str(all))

all = 0
for entity in categories:
    all += len(categories[entity])

print("categoryassignments: " + str(all))


