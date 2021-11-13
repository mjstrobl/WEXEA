import re
import os
import json
import multiprocessing
import time
import datetime
from utils import IGNORED_NAMESPACES, create_file_name_and_directory, jsonKeys2int, find_positions_of_all_links_with_regex, create_filename

from stanza.server import CoreNLPClient, StartServer

ARTICLE_OUTPUTPATH = "articles_3"

def process_article(text,
                    title,
                    title_id,
                    title2Id,
                    redirects,
                    redirects_reverse,
                    aliases_reverse,
                    most_popular_entities,
                    disambiguations_human,
                    disambiguations_geo,
                    links,
                    client, props, annotators):

    article_aliases = {}
    article_aliases_list = []

    if title in aliases_reverse:
        for alias in aliases_reverse[title]:
            if alias[0].islower():
                continue
            alias = alias[:-2] if alias.endswith("'s") else alias
            article_aliases[alias] = {title:1}
            reg = re.escape(alias)
            article_aliases_list.append((alias,re.compile(rf'\b({reg})\b')))

    if title in redirects_reverse:
        for redirect in redirects_reverse[title]:
            if len(redirect) == 0 or redirect[0].islower():
                continue
            redirect = redirect[:-2] if redirect.endswith("'s") else redirect
            if redirect not in article_aliases:
                article_aliases[redirect] = {title:1}
                reg = re.escape(redirect)
                article_aliases_list.append((redirect,re.compile(rf'\b({reg})\b')))

    article_aliases_list.sort(key=lambda x: len(x[0]), reverse=True)


    seen_entities = set()
    complete_content = ''
    for line in text.split('\n'):
        line = line.strip()

        if len(line) == 0:
            continue

        if any(line.lower().startswith('[[' + ignore + ':') for ignore in IGNORED_NAMESPACES):
            continue

        if line.startswith('='):
            complete_content += '\n' + line
            continue

        # find positions of annotations
        line, positions, indices, line_entities = find_positions_of_all_links_with_regex(line, aliases_reverse, redirects_reverse, redirects, article_aliases, article_aliases_list, seen_entities)

        try:
            annotation = client.annotate(line, properties=props, annotators=annotators)
            for i, sent in enumerate(annotation.sentence):
                for mention in sent.mentions:
                    ner = mention.ner
                    tokens = sent.token[mention.tokenStartInSentenceInclusive:mention.tokenEndInSentenceExclusive]
                    start = tokens[0].beginChar
                    end = tokens[-1].endChar
                    alias = line[start:end]
                    if len(indices.intersection(set([j for j in range(start, start + len(alias))]))) == 0:
                        positions.append((start,None,alias,ner))
        except Exception as e:
            print(e)


        positions.sort(key=lambda x: x[0])

        num_additional_characters = 0


        for tuple in positions:
            start = tuple[0] + num_additional_characters
            entity = tuple[1]
            alias = tuple[2]
            tag = tuple[3]

            if entity and tag == 'annotation':
                new_annotation = '[[' + entity + "|" + alias + '|' + tag + ']]'
                #find_positions_of_aliases(title_aliases, aliases_reverse, redirects_reverse, entity)
            elif alias == title or (alias in article_aliases and title in article_aliases[alias]):
                new_annotation = '[[' + title + "|" + alias + '|article_entity]]'
            elif alias in seen_entities:
                new_annotation = '[[' + alias + "|" + alias + '|match_candidate]]'
            elif alias in redirects and redirects[alias] in seen_entities:
                new_annotation = '[[' + redirects[alias] + "|" + alias + '|redirect_candidate]]'
            elif alias in most_popular_entities:
                new_annotation = '[[' + alias + "|" + alias + '|popular_entity]]'
            elif alias in redirects and redirects[alias] in most_popular_entities:
                new_annotation = '[[' + redirects[alias] + "|" + alias + '|popular_redirect]]'
            elif alias in article_aliases and len(article_aliases[alias]) == 1:
                entity = list(article_aliases[alias])[0]
                new_annotation = '[[' + entity + "|" + alias + '|single_candidate]]'
            elif alias in article_aliases:
                entity_or_candidates = article_aliases[alias]
                max_links = 0
                best_candidate = None
                for candidate in entity_or_candidates:
                    if candidate in title2Id:
                        candidate_id = title2Id[candidate]
                        if candidate_id in links:
                            candidate_links = links[candidate_id]
                            if title_id in candidate_links:
                                num_links = candidate_links[title_id]
                                if num_links > max_links:
                                    max_links = num_links
                                    best_candidate = candidate
                if best_candidate:
                    new_annotation = '[[' + best_candidate + "|" + alias + '|best_links]]'
                else:
                    line_candidates = line_entities.intersection(set(entity_or_candidates.keys()))
                    tag = 'best_alias'
                    if len(line_candidates) > 0:
                        tag = 'line_candidate'
                        max_matches = 0
                        for candidate in line_candidates:
                            if entity_or_candidates[candidate] > max_matches:
                                max_matches = entity_or_candidates[candidate]
                                best_candidate = candidate
                    else:
                        max_matches = 0
                        for candidate in entity_or_candidates:
                            if entity_or_candidates[candidate] > max_matches:
                                max_matches = entity_or_candidates[candidate]
                                best_candidate = candidate

                    new_annotation = '[[' + best_candidate + "|" + alias + '|' + tag + ']]'
            else:
                if alias in most_popular_entities:
                    new_annotation = '[[' + alias + "|" + alias + '|popular_entity]]'
                elif alias in redirects and redirects[alias] in most_popular_entities:
                    new_annotation = '[[' + redirects[alias] + "|" + alias + '|popular_redirect]]'
                elif alias in disambiguations_human:
                    new_annotation = '[[' + alias + "|" + alias + '|human_disambiguation]]'
                elif alias in disambiguations_geo:
                    new_annotation = '[[' + alias + "|" + alias + '|geo_disambiguation]]'
                else:
                    new_annotation = '[[' + alias + '|' + tag + ']]'

            num_additional_characters += (len(new_annotation) - len(alias))
            line = line[:start] + new_annotation + line[start + len(alias):]

        complete_content += '\n' + line

    filename = create_file_name_and_directory(title, outputpath + ARTICLE_OUTPUTPATH + '/')
    with open(filename, 'w') as f:
        f.write(complete_content.strip())


def process_articles(process_index,
                     num_processes,
                     title2Id,
                     filename2title,
                     filenames,
                     redirects,
                     redirects_reverse,
                     aliases_reverse,
                     most_popular_entities,
                     disambiguations_human,
                     disambiguations_geo,
                     links, logging_path):

    start_time = time.time()

    counter_all = 0

    new_filename2title = {}

    props = {"ssplit.isOneSentence": True, "ner.applyNumericClassifiers": False,
             "ner.model": "edu/stanford/nlp/models/ner/english.conll.4class.distsim.crf.ser.gz",
             "ner.applyFineGrained": False, "ner.statisticalOnly": True, "ner.useSUTime": False}
    annotators = ['tokenize', 'ssplit', 'pos', 'lemma', 'ner']
    client = CoreNLPClient(
        annotators=annotators,
        properties=props,
        timeout=60000, endpoint="http://localhost:9000", start_server=StartServer.DONT_START, memory='16g')

    logger = open(logging_path + "process_" + str(process_index) + "_logger.txt",'w')

    for i in range(len(filenames)):
        if i % num_processes == process_index:
            filename = filenames[i]
            title = filename2title[filename]
            #if title == "Queen Victoria" or title == "Wilhelm II, German Emperor":
            try:
                if title in title2Id:
                    title_id = title2Id[title]



                    new_filename, _, _, _ = create_filename(title, outputpath + ARTICLE_OUTPUTPATH + '/')
                    new_filename2title[new_filename] = title

                    logger.write("Start with file: " + new_filename + "\n")

                    if not os.path.isfile(new_filename):
                        with open(filename) as f:
                            text = f.read()
                            process_article(text,
                                            title,
                                            title_id,
                                            title2Id,
                                            redirects,
                                            redirects_reverse,
                                            aliases_reverse,
                                            most_popular_entities,
                                            disambiguations_human,
                                            disambiguations_geo,
                                            links,
                                            client, props, annotators)

                        logger.write("File done: " + new_filename + "\n")
                    else:
                        logger.write("File exists: " + new_filename + "\n")

                    counter_all += 1
                    if process_index == 0:
                        time_per_article = (time.time() - start_time) / counter_all
                        print("Process " + str(process_index) + ', articles: ' + str(counter_all) + ", avg time: " +
                              str(time_per_article), end='\r')
            except Exception as e:
                print(e)


    print("Process " + str(process_index) + ', articles processed: ' + str(counter_all))

    with open(dictionarypath + str(process_index) + '_filename2title_3.json', 'w') as f:
        json.dump(new_filename2title, f)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Process " + str(process_index) + ", elapsed time: %s" % str(datetime.timedelta(seconds=elapsed_time)))

    logger.close()

def test():
    config = json.load(open('../config/config.json'))
    outputpath = config['outputpath']
    dictionarypath = outputpath + 'dictionaries/'

    redirects = json.load(open(dictionarypath + 'redirects_pruned.json'))
    redirects_reverse = json.load(open(dictionarypath + 'redirects_reverse.json'))
    aliases_reverse = json.load(open(dictionarypath + 'aliases_reverse.json'))
    most_popular_entities = json.load(open(dictionarypath + 'most_popular_entities.json'))
    disambiguations_human = json.load(open(dictionarypath + 'disambiguations_human.json'))
    disambiguations_geo = json.load(open(dictionarypath + 'disambiguations_geo.json'))
    title2Id = json.load(open(dictionarypath + 'title2Id_pruned.json'))
    links = json.load(open(dictionarypath + 'links_pruned.json'), object_hook=jsonKeys2int)

    props = {"ssplit.isOneSentence": True, "ner.applyNumericClassifiers": False,
             "ner.model": "edu/stanford/nlp/models/ner/english.conll.4class.distsim.crf.ser.gz",
             "ner.applyFineGrained": False, "ner.statisticalOnly": True, "ner.useSUTime": False}
    annotators = ['tokenize', 'ssplit', 'pos', 'lemma', 'ner']
    client = CoreNLPClient(
        annotators=annotators,
        properties=props,
        timeout=60000, endpoint="http://localhost:9000", start_server=False, memory='16g')
    with open('/media/michi/Data/latest_wiki/articles_2/t/te/tex/Textile_(markup_language).txt') as f:
        text = f.read()
        title = 'Textile (markup language)'
        if title in title2Id:
            title_id = title2Id[title]
            print('start processing')
            process_article(text,
                            title,
                            title_id,
                            title2Id,
                            redirects,
                            redirects_reverse,
                            aliases_reverse,
                            most_popular_entities,
                            disambiguations_human,
                            disambiguations_geo,
                            links,
                            client, props, annotators)

if (__name__ == "__main__"):
    config = json.load(open('config/config.json'))
    num_processes = config['processes']
    wikipath = config['wikipath']
    outputpath = config['outputpath']
    logging_path = config['logging_path']
    dictionarypath = outputpath + 'dictionaries/'
    articlepath = outputpath + ARTICLE_OUTPUTPATH + '/'
    try:
        mode = 0o755
        os.mkdir(articlepath, mode)
    except OSError:
        print("directories exist already")

    redirects = json.load(open(dictionarypath + 'redirects_pruned.json'))
    redirects_reverse = json.load(open(dictionarypath + 'redirects_reverse.json'))
    aliases_reverse = json.load(open(dictionarypath + 'aliases_reverse.json'))
    most_popular_entities = json.load(open(dictionarypath + 'most_popular_entities.json'))
    #entities_sorted = json.load(open(dictionarypath + 'entities_sorted.json'))
    disambiguations_human = json.load(open(dictionarypath + 'disambiguations_human.json'))
    disambiguations_geo = json.load(open(dictionarypath + 'disambiguations_geo.json'))
    title2Id = json.load(open(dictionarypath + 'title2Id_pruned.json'))
    links = json.load(open(dictionarypath + 'links_pruned.json'), object_hook=jsonKeys2int)
    filename2title = json.load(open(dictionarypath + 'filename2title_2.json'))
    filenames = list(filename2title.keys())

    print("Read dictionaries.")

    jobs = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=process_articles, args=(i,
                                                                   num_processes,
                                                                   title2Id,
                                                                   filename2title,
                                                                   filenames,
                                                                   redirects,
                                                                   redirects_reverse,
                                                                   aliases_reverse,
                                                                   most_popular_entities,
                                                                   disambiguations_human,
                                                                   disambiguations_geo,
                                                                   links,logging_path))

        jobs.append(p)
        p.start()

    del title2Id
    del filename2title
    del redirects
    del filenames
    del redirects_reverse
    del aliases_reverse
    del most_popular_entities
    del disambiguations_geo
    del disambiguations_human
    del links
    #del entities_sorted

    for job in jobs:
        job.join()