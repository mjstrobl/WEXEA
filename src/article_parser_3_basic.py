import os
import sys
import json
import glob
import time

def process(article,aliases,output_filename):

    with open(article) as f_in:
        with open(output_filename,'w') as f_out:
            for line in f_in:
                if '###' in line:
                    line = line.strip()
                    idx = 0
                    while True:
                        start = line[idx:].find('[[')
                        end = line[idx:].find(']]')
                        if start > -1 and end > -1 and start < end:
                            start += idx
                            end += idx
                            link = line[start + 2:end]
                            tokens = link.split('|')
                            entity = tokens[0]
                            idx = end + 2
                            if len(tokens) > 2 and '###' in entity:
                                alias = tokens[1]
                                candidates = []
                                if alias in aliases:
                                    for candidate in entity.split('###'):
                                        if candidate in aliases[alias]['dict']:
                                            candidates.append((candidate,aliases[alias]['dict'][candidate]))

                                rest = line[end + 2:]
                                if len(candidates) > 0:
                                    candidates.sort(key=lambda tup: tup[1], reverse=True)
                                    line = line[:start] + '[[' + candidates[0][0] + "|" + alias + '|' + tokens[2] + ']]'
                                else:
                                    line = line[:start] + alias

                                idx = len(line)

                                line += rest

                        else:
                            break

                f_out.write(line + '\n')


def main():
    config = json.load(open('config/config.json'))

    outputpath = config['outputpath']
    processed_articlepath = outputpath + 'processed_articles/'
    el_articlepath = outputpath + 'el_articles/'
    aliases = json.load(open(outputpath + 'dictionaries/aliases_sorted_pruned_upper.json'))

    if not os.path.isdir(el_articlepath):
        try:
            mode = 0o755
            os.mkdir(el_articlepath, mode)
        except FileNotFoundError:
            print('Not found: ' + el_articlepath)
            exit(1)


        diff_acc = 0.0
        c = 0

        article_directories = glob.glob(processed_articlepath + "*/")
        for article_directory in article_directories:
            articles = glob.glob(article_directory + "*.txt")

            folder_name = article_directory.split('/')[-2]
            file_directory = el_articlepath + folder_name + '/'

            if not os.path.isdir(file_directory):
                try:
                    mode = 0o755
                    os.mkdir(file_directory, mode)
                except FileNotFoundError:
                    print('Not found: ' + file_directory)
                    exit(1)

            start = time.time()

            for article in articles:
                output_filename = file_directory + article.split('/')[-1]
                process(article,aliases,output_filename)

            end = time.time()
            diff_acc += end - start
            c += len(articles)
            avg = (diff_acc / c) * 1000
            print(str(c) + ', avg t: ' + str(avg), end='\r')

    sys.exit()

if __name__ == '__main__':
    main()
