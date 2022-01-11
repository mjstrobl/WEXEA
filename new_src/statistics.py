import re
import os
import json
import time
import datetime
from utils import RE_LINKS

ner_tags = {"DATE","NUMBER","MISC","LOCATION","PERSON","ORGANIZATION",'DURATION','MONEY','PERCENT','TIME'}

def process_article(text, all_tags):
    for line in text.split('\n'):
        line = line.strip()

        while True:
            match = re.search(RE_LINKS, line)
            if match:
                end = match.end()
                entity = match.group(1)
                parts = entity.split('|')
                tag = parts[-1]

                if len(parts) == 3 or tag in ner_tags:
                    if tag not in all_tags:
                        all_tags[tag] = 0

                    all_tags[tag] += 1
                else:
                    print(entity)

                line = line[end:]
            else:
                break

def process_articles(filenames):
    start_time = time.time()

    print('start processing')

    counter_all = 0

    all_tags = {}

    for i in range(len(filenames)):
        filename = filenames[i]

        with open(filename) as f:
            text = f.read()
            process_article(text,all_tags)

        counter_all += 1
        if counter_all % 1000 == 0:
            time_per_article = (time.time() - start_time) / counter_all
            print("articles: " + str(counter_all) + ", avg time: " + str(time_per_article), end='\r')

    print("articles processed: " + str(counter_all))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("elapsed time: %s" % str(datetime.timedelta(seconds=elapsed_time)))

    print("all tags statistics:")
    for tag in all_tags:
        print(tag + "\t" + str(all_tags[tag]))


if (__name__ == "__main__"):
    config = json.load(open('config/config.json'))
    num_processes = config['processes']
    wikipath = config['wikipath']
    outputpath = config['outputpath']
    logging_path = config['logging_path']
    dictionarypath = outputpath + 'dictionaries/'

    language = 'en'
    if 'language' in config:
        language = config['language']

    print("LANGUAGE: " + language)

    filename2title = json.load(open(dictionarypath + 'filename2title_final.json'))
    filenames = list(filename2title.keys())

    process_articles(filenames)
