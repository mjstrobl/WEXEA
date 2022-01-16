import re
import os
import json
import time
import datetime
import gender_guesser.detector as gender
from utils import create_file_name_and_directory, create_filename, RE_LINKS

ARTICLE_OUTPUTPATH = "articles_final"
NER_TAGS = {"SET", "ORDINAL", "PER", "LOC", "ORG", "DATE","NUMBER","MISC","LOCATION","PERSON","ORGANIZATION",'DURATION','MONEY','PERCENT','TIME'}


def process_article(text, title, corefs, aliases_reverse, coref_assignments):

    current_corefs = {}
    complete_content = ''
    for line in text.split('\n'):
        line = line.strip()

        if len(line) == 0:
            complete_content += '\n'
            continue

        if line.startswith('='):
            complete_content += '\n' + line
            continue

        positions = []
        el_idxs = []
        found_some_corefs = False
        previous_end = 0
        ignore_line = False
        while True:
            match = re.search(RE_LINKS, line)
            if match:
                start = match.start()
                end = match.end()
                entity = match.group(1)
                parts = entity.split('|')
                if (len(parts) != 3 and parts[-1] not in NER_TAGS) or '[[' in entity or ']]' in entity:
                    #TODO: investigate why this happens, it could be a thumbnail in the middle or end of a paragraph of text. ignore it for now.
                    ignore_line = True
                    break
                alias = parts[-2]



                if ("multiple" in parts[-1] or 'part_of_seen_entity' in parts[-1]) and '###' in parts[0]:
                    # use entity linker, which we will deal with later
                    el_idxs.append(len(positions))
                elif 'PERSON' in parts[-1] and parts[0].lower() in coref_assignments and coref_assignments[parts[0].lower()] in current_corefs:
                    # found a coref of a person
                    parts = [current_corefs[coref_assignments[parts[0].lower()]], alias, 'PERSON_coref']
                elif parts[0] in corefs:
                    for c in corefs[parts[0]]:
                        c = c.strip()
                        if c == 'he':
                            c = 'male'
                        elif c == 'she':
                            c = 'female'
                        current_corefs[c] = parts[0]
                elif "PERSON" in parts[-1]:
                    tokens = parts[0].split(' ')
                    types = {'male': 0, 'female': 0, 'unknown': 0, 'andy': 0}
                    for token in tokens:
                        t = gender_detector.get_gender(token)
                        if t.startswith('mostly_'):
                            t = t[7:]
                        types[t] += 1

                    if types['male'] > 0 and types['female'] == 0:
                        current_corefs['male'] = parts[0]
                    elif types['male'] == 0 and types['female'] > 0:
                        current_corefs['female'] = parts[0]

                positions.append((start, len(alias), parts))

                text_in_between = line[previous_end:start].lower()
                line = line[:start] + alias + line[end:]

                for c in current_corefs:
                    if c != 'male' and c != 'female':
                        current_idx = 0
                        while True:
                            idx = text_in_between[current_idx:].find(c)
                            if idx > -1:
                                found_some_corefs = True
                                start = idx + current_idx
                                entity = current_corefs[c]
                                positions.append((start + previous_end, len(c), [entity, c, 'non_person_coref']))
                                current_idx = start + len(c)
                            else:
                                break

                previous_end = match.start() + len(alias)

            else:
                text_in_between = line[previous_end:].lower()
                for c in current_corefs:
                    if c != 'male' and c != 'female':
                        current_idx = 0
                        while True:
                            idx = text_in_between[current_idx:].find(c)
                            if idx > -1:
                                found_some_corefs = True
                                start = idx + current_idx
                                entity = current_corefs[c]
                                positions.append((start + previous_end, len(c), [entity, c, 'non_person_coref']))
                                current_idx = start + len(c)
                            else:
                                break
                break


        if ignore_line:
            continue

        # disambiguate here!
        if len(el_idxs) > 0:
            el_text = line

            disambiguated = False

            if not disambiguated:
                for i in reversed(range(len(el_idxs))):
                    position = positions[el_idxs[i]]
                    start = position[0]
                    length = position[1]
                    parts = position[2]
                    alias = parts[-2]
                    candidates = parts[0].split('###')

                    best_candidate = candidates[0]
                    max_matches = 0
                    for candidate in candidates:
                        if candidate in aliases_reverse and alias in aliases_reverse[candidate] and \
                                aliases_reverse[candidate][alias] > max_matches:
                            max_matches = aliases_reverse[candidate][alias]
                            best_candidate = candidate
                    entity = best_candidate
                    parts = [entity, alias, parts[-1]]

                    positions[el_idxs[i]] = (start, length, [entity, alias, parts[-1]])


        # sort positions and then resolve if found some non person corefs
        if found_some_corefs:
            positions.sort(key=lambda x: x[0])


        for i in reversed(range(len(positions))):
            tuple = positions[i]
            start = tuple[0]
            length = tuple[1]
            annotation = '|'.join(tuple[2])

            line = line[:start] + '[[' + annotation + ']]' + line[start + length:]

        complete_content += '\n' + line

    filename = create_file_name_and_directory(title, outputpath + ARTICLE_OUTPUTPATH + '/')
    with open(filename, 'w') as f:
        f.write(complete_content.strip())


def process_articles(filename2title,filenames, logging_path, corefs, aliases_reverse, coref_assignments):
    start_time = time.time()

    print('start processing')

    counter_all = 0

    new_filename2title = {}

    logger = open(logging_path + "process_logger.txt", 'w')

    for i in range(len(filenames)):
        filename = filenames[i]
        title = filename2title[filename]
        new_filename, _, _, _ = create_filename(title, outputpath + ARTICLE_OUTPUTPATH + '/')
        new_filename2title[new_filename] = title

        logger.write("Start with file: " + new_filename + "\n")

        if not os.path.isfile(new_filename):
            with open(filename) as f:
                text = f.read()
                process_article(text,title,corefs, aliases_reverse, coref_assignments)

            logger.write("File done: " + new_filename + "\n")
        else:
            logger.write("File exists: " + new_filename + "\n")

        counter_all += 1
        if counter_all % 1000 == 0:
            time_per_article = (time.time() - start_time) / counter_all
            print("articles: " + str(counter_all) + ", avg time: " + str(time_per_article), end='\r')

    print("articles processed: " + str(counter_all))

    with open(dictionarypath + 'filename2title_final.json', 'w') as f:
        json.dump(new_filename2title, f)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("elapsed time: %s" % str(datetime.timedelta(seconds=elapsed_time)))

    logger.close()


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

    language = 'en'
    if 'language' in config:
        language = config['language']

    print("LANGUAGE: " + language)

    if language == 'de':
        coref_assignments = {'er': 'male', 'ihn': 'male', 'ihm': 'male', 'sein': 'male', 'seine': 'male', 'seiner': 'male', 'seinem': 'male', 'seinen': 'male', 'seines': 'male',
                             'sie': 'female', 'ihr': 'female', 'ihre': 'female', 'ihrem': 'female', 'ihres': 'female', 'ihrer': 'female'}
    elif language == 'fr':
        coref_assignments = {'il': 'male', 'lui': 'male', 'elle': 'female'}
    else:
        coref_assignments = {'he': 'male', 'his': 'male', 'him': 'male', 'himself': 'male', 'she': 'female', 'her': 'female', 'herself': 'female'}

    filename2title = json.load(open(dictionarypath + 'filename2title_3.json'))
    filenames = list(filename2title.keys())
    aliases_reverse = json.load(open(dictionarypath + 'aliases_reverse.json'))

    corefs = json.load(open('data/corefs.json'))
    gender_detector = gender.Detector()

    print("Read dictionaries.")


    process_articles(filename2title,filenames, logging_path, corefs, aliases_reverse, coref_assignments)
