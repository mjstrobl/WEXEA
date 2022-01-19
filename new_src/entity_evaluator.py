import json
import random
from utils import RE_LINKS
import re

dictionarypath = "/media/michi/Data/wexea/final/en/dictionaries/"

filename2title = json.load(open(dictionarypath + 'filename2title_final.json'))
filenames = list(filename2title.keys())


random.seed(10)
random.shuffle(filenames)

with open('entity_evaluation_2.txt','w') as f:
    counter = 0
    for i in range(20):
        new_filename = filenames[i].replace('articles_final','en/articles')
        with open(new_filename) as wiki:
            title = filename2title[filenames[i]]
            f.write(title + '\n\n')

            for line in wiki:
                printing = False
                original_line = line
                while True:
                    match = re.search(RE_LINKS, line)
                    if match:
                        end = match.end()

                        entity = match.group(1)
                        parts = entity.split('|')
                        entity = parts[0]
                        tag = parts[-1]
                        if entity == title:
                            printing = True
                            break
                        line = line[end:]
                    else:
                        break

                if printing or 'popular_' in original_line or 'single_' in original_line or 'multi_' in original_line or 'coref' in original_line:
                    counter += 1
                    f.write(original_line + '\n')

            f.write('\n\n\n\n')


    print(counter)