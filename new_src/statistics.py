import re
import os
import json
import time
import datetime
from utils import RE_LINKS

ner_tags = {"SET", "ORDINAL", "PER", "LOC", "ORG", "DATE","NUMBER","MISC","LOCATION","PERSON","ORGANIZATION",'DURATION','MONEY','PERCENT','TIME'}

def process_article(text, all_tags, title):
    for line in text.split('\n'):
        line = line.strip()

        while True:
            match = re.search(RE_LINKS, line)
            if match:
                end = match.end()
                entity = match.group(1)
                parts = entity.split('|')
                tag = parts[-1]

                if tag not in all_tags:
                    all_tags[tag] = 0
                    
                if parts[0] == title and tag == 'annotation':
                    all_tags['title'] += 1
                else:
                    all_tags[tag] += 1
                all_tags['total'] += 1

                line = line[end:]
            else:
                break

def process_articles(filenames, filename2title):
    start_time = time.time()

    print('start processing')

    counter_all = 0

    all_tags = {'total':0, 'title': 0}

    for i in range(len(filenames)):
        filename = filenames[i]
        title = filename2title[filename]
        #filename = filename.replace("articles_final","articles_2")
        with open(filename) as f:
            text = f.read()
            process_article(text,all_tags,title)

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


    dictionarypath = "/local/melco2/mstrobl/wexea/final/fr/dictionaries/"

    filename2title = json.load(open(dictionarypath + 'filename2title_final.json'))
    filenames = list(filename2title.keys())

    process_articles(filenames,filename2title)
