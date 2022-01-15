import re
import json
import time
import datetime
from utils import RE_LINKS

ner_tags = {"SET", "ORDINAL", "PER", "LOC", "ORG", "DATE","NUMBER","MISC","LOCATION","PERSON","ORGANIZATION",'DURATION','MONEY','PERCENT','TIME'}

def process_article(text, counters, title):
    counters['articles'] += 1
    for line in text.split('\n'):
        line = line.strip()
        counters['lines'] += 1

        while True:
            match = re.search(RE_LINKS, line)
            if match:
                end = match.end()

                entity = match.group(1)
                parts = entity.split('|')
                entity = parts[0]
                tag = parts[-1]
                if entity != title and tag == "annotation":
                    counters['mentions'] += 1
                line = line[end:]
            else:
                break

def process_articles(filenames,filename2title):
    start_time = time.time()

    print('start processing')

    counter_all = 0

    counters = {'mentions':0, 'articles':0, 'lines':0}

    for i in range(len(filenames)):
        filename = filenames[i]
        title = filename2title[filename]
        #filename = filename.replace("articles_final","articles_2")
        try:
            with open(filename) as f:
                text = f.read()
                process_article(text,counters,title)

            counter_all += 1
            if counter_all % 1000 == 0:
                time_per_article = (time.time() - start_time) / counter_all
                print("articles: " + str(counter_all) + ", avg time: " + str(time_per_article), end='\r')
        except:
            n = 0

    print("articles processed: " + str(counter_all))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("elapsed time: %s" % str(datetime.timedelta(seconds=elapsed_time)))

    print("counters statistics:")
    for tag in counters:
        print(tag + "\t" + str(counters[tag]))


if (__name__ == "__main__"):


    dictionarypath = "/local/melco2/mstrobl/wexea/final/es/dictionaries/"

    filename2title = json.load(open(dictionarypath + 'filename2title_final.json'))
    filenames = list(filename2title.keys())

    process_articles(filenames,filename2title)

