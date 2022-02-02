import json
import os
import re

language = 'de'

RE_LINKS = re.compile(r'\[{2}(.*?)\]{2}', re.DOTALL | re.UNICODE)

dbpedia_data_filename = "/media/michi/Data/datasets/dbpedia/infobox-properties_lang=" + language + ".ttl"
dbpedia_labels_filename = "/media/michi/Data/datasets/dbpedia/labels_lang=" + language + ".ttl"
dbpedia_ids_filename = "/media/michi/Data/datasets/dbpedia/page_lang=" + language + "_ids.ttl"
id2title = json.load(open('/media/michi/Data/wexea/final/' + language + '/dictionaries/id2title.json'))

def getRelevantPairs():
    relevant_pairs_filename = '/media/michi/Data/wikipedia/relevant_pairs_dbpedia_' + language + '.json'
    if not os.path.isfile(relevant_pairs_filename):
        dbpedia2title = {}
        with open(dbpedia_ids_filename) as f:
            for line in f:
                tokens = line.strip().split(' ')
                if len(tokens) == 4:
                    subject = tokens[0]
                    wiki_id = tokens[2].split('"')[1]
                    if wiki_id in id2title:
                        dbpedia2title[subject] = id2title[wiki_id]

        print("Found " + str(len(dbpedia2title)) + " titles.")

        relevant_pairs = {}
        with open(dbpedia_data_filename) as f:
            for line in f:
                tokens = line.strip().split(' ')
                if len(tokens) == 4:
                    subject = tokens[0]
                    relation = tokens[1]
                    object = tokens[2]
                    if subject in dbpedia2title and object in dbpedia2title:
                        l = [dbpedia2title[subject],dbpedia2title[object]]
                        l.sort()
                        pair = '###'.join(l)
                        if pair not in relevant_pairs:
                            relevant_pairs[pair] = []
                        relevant_pairs[pair].append(relation)

        print(str(len(relevant_pairs)) + ' relevant pairs.')

        with open(relevant_pairs_filename,'w') as f:
            json.dump(relevant_pairs,f)

        return relevant_pairs
    else:
        relevant_pairs = json.load(open(relevant_pairs_filename))
        return relevant_pairs

def find_wexea_pairs():
    wexea_pairs_filename = '/media/michi/Data/wikipedia/wexea_pairs_dbpedia_' + language + '.json'
    if not os.path.isfile(wexea_pairs_filename):
        wexea_pairs = {}

        filename2title = json.load(open('/media/michi/Data/wexea/final/en/dictionaries/filename2title_final.json'))
        filenames = list(filename2title.keys())
        counter = 0
        for filename in filenames:
            counter += 1
            if os.path.isfile(filename):
                with open(filename) as f:
                    for line in f:
                        line = line.strip()
                        entities = set()
                        while True:
                            match = re.search(RE_LINKS, line)
                            if match:
                                end = match.end()
                                entity = match.group(1)
                                parts = entity.split('|')
                                entity = parts[0]
                                entities.add(entity)
                                line = line[end:]
                            else:
                                break

                        entities_list = list(entities)
                        entities_list.sort()
                        for i in range(len(entities_list)):
                            for j in range(i, len(entities_list)):
                                if i != j:
                                    key = entities_list[i] + '###' + entities_list[j]
                                    if key not in wexea_pairs:
                                        wexea_pairs[key] = 0
                                    wexea_pairs[key] += 1

            if counter % 1000 == 0:
                print("Wexea processed: " + str(counter), end='\r')

        print()
        print('found ' + str(len(wexea_pairs)) + ' pairs.')
        with open(wexea_pairs_filename, 'w') as f:
            json.dump(wexea_pairs, f)

        return wexea_pairs
    else:
        wexea_pairs = json.load(open(wexea_pairs_filename))
        return wexea_pairs

def find_wiki_pairs():
    wiki_pairs_filename = '/media/michi/Data/wikipedia/wiki_pairs_dbpedia_' + language + '.json'
    if not os.path.isfile(wiki_pairs_filename):
        wiki_pairs = {}

        filename2title = json.load(open('/media/michi/Data/wexea/final/' + language + '/dictionaries/filename2title_final.json'))
        filenames = list(filename2title.keys())
        counter = 0
        for filename in filenames:
            counter += 1
            title = filename2title[filename]
            if os.path.isfile(filename):
                with open(filename) as f:
                    for line in f:
                        line = line.strip()
                        entities = set()
                        while True:
                            match = re.search(RE_LINKS, line)
                            if match:
                                end = match.end()
                                entity = match.group(1)
                                parts = entity.split('|')
                                entity = parts[0]

                                if entity != title and parts[-1] == 'annotation':
                                    entities.add(entity)
                                line = line[end:]
                            else:
                                break

                        entities_list = list(entities)
                        entities_list.sort()
                        for i in range(len(entities_list)):
                            for j in range(i, len(entities_list)):
                                if i != j:
                                    key = entities_list[i] + '###' + entities_list[j]
                                    if key not in wiki_pairs:
                                        wiki_pairs[key] = 0
                                    wiki_pairs[key] += 1

            if counter % 1000 == 0:
                print("Wiki processed: " + str(counter),end='\r')

        print()

        print('found ' + str(len(wiki_pairs)) + ' pairs.')
        with open(wiki_pairs_filename, 'w') as f:
            json.dump(wiki_pairs, f)

        return wiki_pairs
    else:
        wiki_pairs = json.load(open(wiki_pairs_filename))
        return wiki_pairs

def find(type,w_pairs,relevant_pairs):
    relation_matches = {}
    relation_matches_unique = {}

    for pair in w_pairs:
        if pair in relevant_pairs:
            c = w_pairs[pair]
            relations = relevant_pairs[pair]
            for relation in relations:
                if relation not in relation_matches:
                    relation_matches[relation] = 0
                relation_matches[relation] += c

    all_relations = set()
    for relation in relation_matches:
        if relation_matches[relation] >= 100:
            all_relations.add(relation)

    annotations = 0
    pairs = 0
    relation_matches = {}
    for pair in w_pairs:
        if pair in relevant_pairs:
            c = w_pairs[pair]

            relations = relevant_pairs[pair]
            consider_pair = False
            for relation in relations:
                if relation in all_relations:
                    consider_pair = True
                    if relation not in relation_matches_unique:
                        relation_matches_unique[relation] = set()
                    relation_matches_unique[relation].add(pair)
                    if relation not in relation_matches:
                        relation_matches[relation] = 0
                    relation_matches[relation] += c
                    annotations += c
            if consider_pair:
                pairs += c

    print('Annotations including all relations: ' + str(annotations))
    print('Pairs found: ' + str(pairs))

    with open('/media/michi/Data/wikipedia/statistic_data/data_' + type + '_' + language + '.json', 'w') as f:
        json.dump(relation_matches, f)

    new_relation_matches_unique = {}
    for relation in relation_matches_unique:
        new_relation_matches_unique[relation] = list(relation_matches_unique[relation])

    with open('/media/michi/Data/wikipedia/statistic_data/data_unique_' + type + '_' + language + '.json', 'w') as f:
        json.dump(new_relation_matches_unique, f)

    data_unique = new_relation_matches_unique

    data = relation_matches

    avg = 0
    avg_sentences_per_relation = 0

    for relation in data:
        avg_sentences_per_relation += data[relation]

    avg_sentences_per_relation /= len(data)

    l_data_unique = []
    for relation in data_unique:
        l_data_unique.append((relation, len(data_unique[relation])))
        avg += len(data_unique[relation])

    avg = avg / len(data_unique)

    l_data_unique.sort(key=lambda x: x[1], reverse=True)

    # print(l_data_unique[:10])
    print(type)
    print('Avg per relation of unique entity pairs: ' + str(avg))
    print('Avg per relation of sentences to be extracted: ' + str(avg_sentences_per_relation))
    print('Num relations: ' + str(len(data)))
    print()
    print()



def main():
    wexea_pairs = find_wexea_pairs()
    print('found wexea pairs: ' + str(len(wexea_pairs)))
    wiki_pairs = find_wiki_pairs()
    print('found wiki pairs: ' + str(len(wiki_pairs)))

    relevant_pairs = getRelevantPairs()
    print('found relevant pairs: ' + str(len(relevant_pairs)))

    find('wexea', wexea_pairs, relevant_pairs)
    find('wiki', wiki_pairs, relevant_pairs)

if (__name__ == "__main__"):
    main()