import re
import os
import json
import multiprocessing
import time
import datetime
from utils import create_file_name_and_directory, jsonKeys2int, find_positions_of_all_links_with_regex, create_filename, RE_LINKS

from stanza.server import CoreNLPClient, StartServer

from language_variables import IGNORED_NAMESPACES

ARTICLE_OUTPUTPATH = "articles_3"

def process_article(text,
                    title,
                    title_id,
                    title2Id,
                    redirects,
                    redirects_reverse,
                    aliases_reverse,
                    most_popular_entities,
                    persons,
                    disambiguations_human,
                    disambiguations_geo,
                    links,
                    client,
                    props,
                    annotators):

    article_aliases = {}
    article_aliases_list = []

    if title in aliases_reverse:
        for alias in aliases_reverse[title]:
            if alias[0].islower() or alias.isnumeric():
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

    acronyms = {}
    seen_entities = set()
    seen_entities_split = {}
    complete_content = ''
    for line in text.split('\n'):
        line = line.strip()

        if len(line) == 0:
            complete_content += '\n'
            continue

        if any(line.lower().startswith('[[' + ignore + ':') for ignore in IGNORED_NAMESPACES):
            continue

        if line.startswith('='):
            while True:
                match = re.search(RE_LINKS, line)
                if match:
                    start = match.start()
                    end = match.end()
                    entity = match.group(1)
                    parts = entity.split('|')
                    alias = parts[-1]
                    line = line[:start] + alias + line[end:]
                else:
                    break
                    
            complete_content += '\n' + line
            continue

        # find positions of annotations
        line, positions, indices, line_entities = find_positions_of_all_links_with_regex(acronyms, line, aliases_reverse, redirects_reverse, redirects, article_aliases, article_aliases_list, seen_entities, seen_entities_split)
        sentence_breaks = []
        try:
            annotation = client.annotate(line, properties=props, annotators=annotators)
            for i, sent in enumerate(annotation.sentence):
                sentence_breaks.append(sent.characterOffsetEnd)
                for mention in sent.mentions:
                    ner = mention.ner
                    tokens = sent.token[mention.tokenStartInSentenceInclusive-sent.tokenOffsetBegin:mention.tokenEndInSentenceExclusive-sent.tokenOffsetBegin]
                    start = tokens[0].beginChar
                    end = tokens[-1].endChar
                    alias = line[start:end]

                    if len(indices.intersection(set([j for j in range(start, start + len(alias))]))) == 0:
                        if alias in acronyms:
                            positions.append((start, acronyms[alias], alias, 'acronym_ner'))
                        else:
                            positions.append((start,None,alias,ner))
        except Exception as e:
            #print(e)
            pass


        positions.sort(key=lambda x: x[0])

        num_additional_characters = 0

        current_sent_idx = 0
        for i in range(len(positions)):
            tuple = positions[i]
            while current_sent_idx < len(sentence_breaks) and tuple[0] > sentence_breaks[current_sent_idx]:
                sentence_breaks[current_sent_idx] += num_additional_characters
                current_sent_idx += 1
            start = tuple[0] + num_additional_characters
            entity = tuple[1]
            alias = tuple[2]
            alias_to_print = alias
            tag = tuple[3]

            tag_to_print = tag

            if alias is not None and len(alias) > 0:
                if tag == 'DATE' or tag == 'TIME' or tag == 'DURATION' or tag == 'SET' or tag == 'MONEY' or tag == 'PERCENT' or tag == 'ORDINAL':
                    tag_to_print = tag
                    entity = None
                elif tag == "acronym_entity":
                    entity = alias
                elif tag == "acronym" and i > 0:
                    entity = positions[i-1][2]
                elif entity and tag == 'annotation':
                    if alias[0].isupper() or alias[0].isnumeric():
                        tag_to_print = tag
                    else:
                        entity = None
                        alias_to_print = None
                elif alias == title or (alias in article_aliases and title in article_aliases[alias]):
                    # alias seems to correspond to article entity
                    entity = title
                    tag_to_print += "_article_entity"
                elif alias in seen_entities:
                    # alias matches already seen entity
                    entity = alias
                    tag_to_print += "_match_candidate"
                elif alias in redirects and redirects[alias] in seen_entities:
                    # alias in redirects and redirected alias matches already seen entity
                    entity = redirects[alias]
                    tag_to_print += "_redirect_candidate"
                elif alias in most_popular_entities:
                    # alias is a popular entity
                    entity = alias
                    tag_to_print += "_popular_entity"
                elif alias in redirects and redirects[alias] in most_popular_entities:
                    # redirect of alias is a popular entity
                    entity = redirects[alias]
                    tag_to_print += "_popular_redirect"
                elif alias in article_aliases and len(article_aliases[alias]) == 1:
                    # alias refers to a single candidate through aliases from all already seen entities
                    entity = list(article_aliases[alias])[0]
                    tag_to_print += "_single_candidate"
                elif alias in article_aliases:
                    # alias has multiple candidates from entities already seen
                    candidates = article_aliases[alias]

                    all_persons = True
                    file_entity_in_candidates = False
                    for candidate in candidates:
                        if candidate not in persons:
                            all_persons = False
                            break
                    max_links = 0
                    best_candidate = None

                    if all_persons:
                        for candidate in candidates:
                            if candidate in title2Id:
                                candidate_id = title2Id[candidate]
                                #candidate_id is an integer, but links contains strings as Ids
                                candidate_id = str(candidate_id)
                                if candidate_id in links:
                                    candidate_links = links[candidate_id]
                                    # title_id needs to be string
                                    title_id = str(title_id)
                                    if title_id in candidate_links:
                                        num_links = candidate_links[title_id]
                                        if num_links > max_links:
                                            max_links = num_links
                                            best_candidate = candidate
                    if best_candidate and all_persons:
                        # all candidates are persons and we pick the one with the most links to the article entity
                        entity = best_candidate
                        tag_to_print += "_best_links"
                    else:
                        # candidates either not all persons or there are no linked candidates, EL has to figure it out.
                        line_candidates = line_entities.intersection(set(candidates.keys()))
                        new_tag = '_best_alias'
                        if len(line_candidates) > 0:
                            new_tag = '_line_candidate'

                            if len(line_candidates) == 1:
                                new_tag = '_single_line_candidate'
                                entity = list(line_candidates)[0]
                            else:
                                new_tag = '_multiple_line_candidates'
                                entity = '###'.join(list(line_candidates))
                        else:
                            new_tag = '_multiple_candidates'
                            entity = '###'.join(list(candidates.keys()))

                        tag_to_print += new_tag
                elif alias in disambiguations_human:
                    # best that could be found is a human disambiguation page
                    entity = alias
                    tag_to_print += "_human_disambiguation"
                elif alias in disambiguations_geo:
                    # best that could be found is a geographical disambiguation page
                    entity = alias
                    tag_to_print += "_geo_disambiguation"
                else:
                    # check if alias part of seen entity
                    if alias in seen_entities_split:
                        entity = '###'.join(seen_entities_split[alias])
                        tag_to_print += "_part_of_seen_entity"

                if tag == "acronym_entity":
                    new_tuple = (start,entity,alias,tag)
                    positions[i] = new_tuple
            else:
                entity = None
                alias_to_print = None

            if entity is None:
                if alias_to_print is None:
                    new_annotation = alias
                    alias_to_print = alias
                else:
                    new_annotation = '[[' + alias_to_print + '|' + tag_to_print + ']]'
            else:
                new_annotation = '[[' + entity + "|" + alias_to_print + '|' + tag_to_print + ']]'

            num_additional_characters += (len(new_annotation) - len(alias_to_print))
            line = line[:start] + new_annotation + line[start + len(alias_to_print):]
        
        for i in reversed(range(len(sentence_breaks)-1)):
            line = line[:sentence_breaks[i]] + '\n' + line[sentence_breaks[i]:].strip()
        
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
                     persons,
                     disambiguations_human,
                     disambiguations_geo,
                     links, logging_path, language):

    start_time = time.time()

    print('start processing')

    counter_all = 0

    new_filename2title = {}


    sutime_path = config['sutime']

    if language == 'de':
        sutime_rules = "edu/stanford/nlp/models/sutime/defs.sutime.txt,edu/stanford/nlp/models/sutime/english.sutime.txt,edu/stanford/nlp/models/sutime/english.holidays.sutime.txt," + sutime_path + "german.sutime.txt"
        props = {"tokenize.language": "de", "pos.model": "edu/stanford/nlp/models/pos-tagger/german-ud.tagger",
                 "tokenize.postProcessor": "edu.stanford.nlp.international.german.process.GermanTokenizerPostProcessor",
                 "ner.applyFineGrained": False,
                 "ner.model": "edu/stanford/nlp/models/ner/german.distsim.crf.ser.gz",
                 "ner.applyNumericClassifiers": True, "ner.useSUTime": True, "ner.language": "fr",
                 "sutime.rules": sutime_rules
                 }
    elif language == 'fr':
        sutime_rules = "edu/stanford/nlp/models/sutime/defs.sutime.txt,edu/stanford/nlp/models/sutime/english.sutime.txt,edu/stanford/nlp/models/sutime/english.holidays.sutime.txt," + sutime_path + "french.sutime.txt"
        props = {"tokenize.language": "fr", "pos.model": "edu/stanford/nlp/models/pos-tagger/french-ud.tagger",
                 "ner.applyFineGrained": False,
                 "ner.model": "edu/stanford/nlp/models/ner/french-wikiner-4class.crf.ser.gz",
                 "ner.applyNumericClassifiers": True, "ner.useSUTime": True, "ner.language": "fr",
                 "sutime.rules": sutime_rules
                 }

    elif language == 'es':
        props = {"tokenize.language": "es", "pos.model": "edu/stanford/nlp/models/pos-tagger/spanish-ud.tagger",
                 "ner.applyFineGrained": False,
                 "ner.model": "edu/stanford/nlp/models/ner/spanish.ancora.distsim.s512.crf.ser.gz",
                 "ner.applyNumericClassifiers": True, "ner.useSUTime": True, "ner.language": "es",
                 "sutime.rules": "edu/stanford/nlp/models/sutime/defs.sutime.txt,edu/stanford/nlp/models/sutime/english.sutime.txt,edu/stanford/nlp/models/sutime/english.holidays.sutime.txt,edu/stanford/nlp/models/sutime/spanish.sutime.txt"
                 }
    else:
        props = {"ner.applyFineGrained": False,
                 "ner.model": "edu/stanford/nlp/models/ner/english.muc.7class.distsim.crf.ser.gz"}

    annotators = ['tokenize', 'ssplit', 'pos', 'lemma', 'ner']
    client = CoreNLPClient(
        annotators=annotators,
        properties=props,
        timeout=60000, endpoint="http://localhost:9000", start_server=StartServer.DONT_START, memory='16g')

    logger = open(logging_path + "process_" + str(process_index) + "_logger.txt", 'w')

    for i in range(len(filenames)):
        if i % num_processes == process_index:
            filename = filenames[i]
            title = filename2title[filename]
            #if title == "Queen Victoria" or title == "Wilhelm II, German Emperor" or title == 'Queen Victoria Park':
            #try:
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
                                        persons,
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
            #except Exception as e:
            #    print(e)
            #    pass

    print("Process " + str(process_index) + ', articles processed: ' + str(counter_all))

    with open(dictionarypath + str(process_index) + '_filename2title_3.json', 'w') as f:
        json.dump(new_filename2title, f)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Process " + str(process_index) + ", elapsed time: %s" % str(datetime.timedelta(seconds=elapsed_time)))

    logger.close()

def merge_all_dictionaries(num_processes, dictionarypath):
    filename2title = {}

    for i in range(num_processes):
        partial_filename2title = json.load(open(dictionarypath + str(i) + '_filename2title_3.json'))
        for filename in partial_filename2title:
            filename2title[filename] = partial_filename2title[filename]

    with open(dictionarypath + 'filename2title_3.json', 'w') as f:
        json.dump(filename2title, f)


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
        os.mkdir(logging_path, mode)
    except OSError:
        print("directories exist already")

    try:
        mode = 0o755
        os.mkdir(articlepath, mode)
    except OSError:
        print("directories exist already")


    language = 'en'
    if 'language' in config:
        language = config['language']

    print("LANGUAGE: " + language)

    redirects = json.load(open(dictionarypath + 'redirects_pruned.json'))
    persons = set(json.load(open('data/persons.json')))
    redirects_reverse = json.load(open(dictionarypath + 'redirects_reverse.json'))
    aliases_reverse = json.load(open(dictionarypath + 'aliases_reverse.json'))
    most_popular_entities = json.load(open(dictionarypath + 'most_popular_entities.json'))
    #entities_sorted = json.load(open(dictionarypath + 'entities_sorted.json'))
    disambiguations_human = json.load(open(dictionarypath + 'disambiguations_human.json'))
    disambiguations_geo = json.load(open(dictionarypath + 'disambiguations_geo.json'))
    title2Id = json.load(open(dictionarypath + 'title2id_pruned.json'))
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
                                                                   persons,
                                                                   disambiguations_human,
                                                                   disambiguations_geo,
                                                                   links,logging_path,language))

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

    merge_all_dictionaries(num_processes, dictionarypath)
