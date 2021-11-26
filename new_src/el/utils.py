import json
import re
import os



RE_LINKS = re.compile(r'\[{2}(.*?)\]{2}', re.DOTALL | re.UNICODE)

config = json.load(open('../../config/config.json'))
outputpath = config['outputpath']

wexea_directory = outputpath
title2filename = json.load(open(wexea_directory + 'dictionaries/title2filename.json'))

def create_filename(title,outputpath):
    file_directory = re.sub('[^0-9a-z]+', '_', title.lower())
    while len(file_directory) < 3:
        file_directory += '_'

    if len(file_directory) > 3:
        file_directory = file_directory[:3]

    first_directory = outputpath + file_directory[:1] + '/'
    second_directory = first_directory + file_directory[:2] + '/'
    third_directory = second_directory + file_directory + '/'

    filename = title.replace(" ", '_').replace('/', '_') + '.txt'
    filename = third_directory + filename

    return filename, first_directory, second_directory, third_directory

def create_file_name_and_directory(title, outputpath):

    filename, first_directory, second_directory, third_directory = create_filename(title,outputpath)

    if create_directory(first_directory) and create_directory(second_directory) and create_directory(third_directory):

        return filename
    else:
        return None

def create_directory(directory):
    if not os.path.isdir(directory):
        try:
            mode = 0o755
            os.mkdir(directory, mode)
        except FileExistsError:
            return True
        except FileNotFoundError:
            print('FileNotFoundError: ' + directory)
            return False

    return True

def start():
    counter = 0
    entity_title2filename = {}
    directorypath = outputpath + 'entities/'
    try:
        mode = 0o755
        os.mkdir(directorypath, mode)
    except OSError:
        print("directories exist already")

    for title in title2filename:
        entities = set()
        with open(title2filename[title]) as f:
            text = f.read()
            matches = re.findall(RE_LINKS, text)
            if matches:
                for match in matches:
                    alias = match
                    entity = alias
                    pos_bar = alias.find('|')
                    if pos_bar > -1:
                        entity = alias[:pos_bar]
                        alias = alias[pos_bar+1:]
                    entities.add(entity)

        if len(entities) > 0:
            filename = create_file_name_and_directory(title, directorypath)
            entity_title2filename[title] = filename
            with open(filename,'w') as f:
                f.write(' '.join(entities) + '\n')

        counter += 1

        if counter % 1000 == 0:
            print('Articles processed: ' + str(counter),end='\r')

    with open(outputpath + 'dictionaries/entity_title2filename.json','w') as f:
        json.dump(entity_title2filename,f)

start()