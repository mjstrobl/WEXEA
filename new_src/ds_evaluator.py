import json

language = 'en'

data_wexea = json.load(open('/media/michi/Data/wikipedia/statistic_data/data_wexea_' + language + '.json'))
data_wiki = json.load(open('/media/michi/Data/wikipedia/statistic_data/data_wiki_' + language + '.json'))


wexea = []
wiki = []

for relation in data_wexea:
    wexea.append((relation,data_wexea[relation]))

for relation in data_wiki:
    wiki.append((relation, data_wiki[relation]))

wexea.sort(key=lambda x:x[1],reverse=True)
wiki.sort(key=lambda x:x[1],reverse=True)

for tuple in wexea:
    relation = tuple[0]
    if relation in data_wiki:
        print(relation + '\t' + str(data_wiki[relation]) + '\t' + str(data_wexea[relation]))

