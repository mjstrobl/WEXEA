from pathlib import Path
import os
from shutil import copyfile
import shutil

dry_run = False

directory = '/media/michi/Data/wexea/final/en/'
language = 'en'

directory_wexea = directory + 'articles_final/'
directory_list_of = directory + 'list_of/'
directory_categories = directory + 'categories/'
directory_disambiguations_geo = directory + 'disambiguations_geo/'
directory_disambiguations_human = directory + 'disambiguations_human/'
directory_disambiguations_number = directory + 'disambiguations_number/'
directory_disambiguations_other = directory + 'disambiguations_other/'


wexea = directory + 'output/' + language + '_wexea.txt'
list_of = directory + 'output/others/' + language + '_list_of.txt'
categories = directory + 'output/others/' + language + '_categories.txt'
disambiguations_geo = directory + 'output/others/' + language + '_disambiguations_geo.txt'
disambiguations_human = directory + 'output/others/' + language + '_disambiguations_human.txt'
disambiguations_number = directory + 'output/others/' + language + '_disambiguations_number.txt'
disambiguations_other = directory + 'output/others/' + language + '_disambiguations_other.txt'

if not dry_run:
    try:
        mode = 0o755
        os.mkdir(directory + 'output/', mode)
        os.mkdir(directory + 'output/dictionaries/', mode)
        os.mkdir(directory + 'output/others/', mode)
    except OSError:
        print("output directory exists already")

dictionaries = [(directory + 'dictionaries/aliases_pruned.json',directory + 'output/dictionaries/aliases.json'),
                (directory + 'dictionaries/aliases_reverse.json',directory + 'output/dictionaries/aliases_reverse.json'),
                (directory + 'dictionaries/categories.json',directory + 'output/dictionaries/categories.json'),
                (directory + 'dictionaries/category_redirects.json',directory + 'output/dictionaries/category_redirects.json'),
                (directory + 'dictionaries/disambiguations_geo.json',directory + 'output/dictionaries/disambiguations_geo.json'),
                (directory + 'dictionaries/disambiguations_human.json',directory + 'output/dictionaries/disambiguations_human.json'),
                (directory + 'dictionaries/disambiguations_number.json',directory + 'output/dictionaries/disambiguations_number.json'),
                (directory + 'dictionaries/disambiguations_other.json',directory + 'output/dictionaries/disambiguations_other.json'),
                (directory + 'dictionaries/given_names.json',directory + 'output/dictionaries/given_names.json'),
                (directory + 'dictionaries/id2title_pruned.json',directory + 'output/dictionaries/id2title.json'),
                (directory + 'dictionaries/links_pruned.json',directory + 'output/dictionaries/links.json'),
                (directory + 'dictionaries/most_popular_entities.json',directory + 'output/dictionaries/most_popular_entities.json'),
                (directory + 'dictionaries/redirects_pruned.json',directory + 'output/dictionaries/redirects.json'),
                (directory + 'dictionaries/surnames.json',directory + 'output/dictionaries/surnames.json'),
                (directory + 'dictionaries/title2id_pruned.json',directory + 'output/dictionaries/title2id.json'),
                (directory + 'dictionaries/id2title_pruned.json',directory + 'output/dictionaries/id2title.json')]

files = [(directory_wexea,wexea),(directory_list_of,list_of),(directory_categories,categories),(directory_disambiguations_geo,disambiguations_geo),
         (directory_disambiguations_human,disambiguations_human),(directory_disambiguations_number,disambiguations_number),(directory_disambiguations_other,disambiguations_other)]

for tuple in files:
    files_directory = tuple[0]
    filename = tuple[1]
    counter = 0
    print("create file: " + filename)
    if not os.path.isfile(filename):
        if not dry_run:
            with open(filename,'w') as f_out:
                for path in Path(files_directory).rglob('*.txt'):
                    f_out.write("###FILENAME### " + path.name + "\n")
                    with open(path) as f:
                        for line in f:
                            f_out.write(line)
                    f_out.write('\n\n\n\n')
                    counter += 1
                    if counter % 1000 == 0:
                        print("Processed: " + str(counter), end='\r')
    else:
        print('file already exists.')

print('All files written.')

for tuple in dictionaries:
    src = tuple[0]
    dst = tuple[1]
    if os.path.isfile(src):
        if os.path.isfile(dst):
            print("File already copied: " + dst)
        else:
            print("copy file src: " + src)
            if not dry_run:
                copyfile(src, dst)
            print("copied file to dst: " + dst)
    else:
        print('File not found: ' + src)


print('zip directory')
zip_filename = directory + language + "_wexea"
if os.path.isfile(zip_filename):
    print("zip file already exists")
elif not dry_run:
    shutil.make_archive(zip_filename, 'zip', directory + 'output/')
print("done")