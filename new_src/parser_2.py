import re
import os
import json
import multiprocessing
import time
import datetime
from utils import is_upper, IGNORED_NAMESPACES, find_categories, find_entities, add_disambgiuation, create_file_name_and_directory, remove_notes, create_filename, find_same_entity_mentions, remove_external_links, clean_text, remove_files, remove_galleries, remove_tables, remove_templates, remove_irrelevant_sections

ARTICLE_OUTPUTPATH = "articles_2"

RE_DISAMBIGUATIONS = '{{set index article}}|{{SIA}}|{{disambiguation\||\|disambiguation}}|{{disambiguation}}|{{disamb}}|{{disambig}}|{{disamb\||\|disamb}}|{{disambig\||\|disambig}}|{{dab\||\|dab}}|{{dab}}|{{disambiguation cleanup}}'
RE_HUMAN_DISAMBIGUATIONS = '{{hndis\||\|hndis}}|{{hndis}}|{{human name disambiguation}}|{{human name disambiguation\||\|human name disambiguation}}'
RE_GEO_DISAMBIGUATIONS = '{{place name disambiguation}}|{{geodis}}|{{geodis\||\|geodis}}'
RE_NUMBER_DISAMBIGUATIONS = '{{number disambiguation\||\|number disambiguation}}|{{numdab\||\|numdab}}|{{numberdis\||\|numberdis}}|{{numberdis}}|{{numdab}}|{{number disambiguation}}'
RE_STUB = 'stub}}'
GIVEN_NAMES = '{{given name}}', '[[Category:Given names]]', '[[Category:Masculine given names]]', '[[Category:Feminine given names]]'
SURNAMES = '{{surname}}', '[[Category:Surnames]]'

def process_list(text, redirects, aliases):
    find_entities(text, redirects, aliases)

def process_article(text,
                    outputpath,
                    title,
                    stubs,
                    categories,
                    human_disambiguations,
                    geo_disambiguations,
                    number_disambiguations,
                    other_disambiguations,
                    redirects, given_names,
                    surnames,
                    title_id,
                    links,
                    aliases,
                    new_filename2title):

    lines = text.split("\n")
    if re.findall(RE_STUB, text.lower()):
        stubs.append(title)
        # stub article
        filename = outputpath + 'stubs/' + title.replace(" ", '_').replace('/', '_') + '.txt'
        with open(filename, 'w') as f:
            f.write(text)
        find_entities(text, redirects, aliases)
    elif any(re.findall(RE_HUMAN_DISAMBIGUATIONS, text.lower())):
        filename = outputpath + 'disambiguations_human/' + title.replace(" ", '_').replace('/', '_') + '.txt'
        with open(filename, 'w') as f:
            f.write(text)
        add_disambgiuation(lines, title, human_disambiguations, redirects, aliases)
    elif any(re.findall(RE_GEO_DISAMBIGUATIONS, text.lower())):
        filename = outputpath + 'disambiguations_geo/' + title.replace(" ", '_').replace('/', '_') + '.txt'
        with open(filename, 'w') as f:
            f.write(text)
        add_disambgiuation(lines, title, geo_disambiguations, redirects, aliases)
    elif any(re.findall(RE_NUMBER_DISAMBIGUATIONS, text.lower())):
        filename = outputpath + 'disambiguations_number/' + title.replace(" ", '_').replace('/', '_') + '.txt'
        with open(filename, 'w') as f:
            f.write(text)
        add_disambgiuation(lines, title, number_disambiguations, redirects, aliases)
    elif any(re.findall(RE_DISAMBIGUATIONS, text.lower())) or '(disambiguation)' in title:
        filename = outputpath + 'disambiguations_other/' + title.replace(" ", '_').replace('/', '_') + '.txt'
        with open(filename, 'w') as f:
            f.write(text)
        add_disambgiuation(lines, title, other_disambiguations, redirects, aliases)
    elif not any(title.lower().startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):

        links[title_id] = {}

        # check if articles is about names
        if any(given_name in text for given_name in GIVEN_NAMES):
            given_names.append(title)
        if any(given_name in text for given_name in SURNAMES):
            surnames.append(title)

        filename = create_file_name_and_directory(title, outputpath + ARTICLE_OUTPUTPATH + '/')

        pos_infobox = text.find('{{Infobox')
        if pos_infobox > -1:
            text = text[pos_infobox:]

        find_entities(text, redirects, aliases, title2Id, title_id, links)
        text = remove_templates(text)
        text = remove_notes(text)
        text = remove_galleries(text)
        text = remove_files(text)
        text = find_categories(text, categories, title)
        text = remove_irrelevant_sections(text)
        text = remove_tables(text)
        text = find_same_entity_mentions(text, title)
        text = clean_text(text)
        text, external_links = remove_external_links(text)

        new_filename2title[filename] = title

        with open(filename, 'w') as f:
            f.write(text + '\n')

def process_articles(process_index, num_processes, outputpath, redirects, title2Id, filenames, filename2title, lists):
    start_time = time.time()

    aliases = {}
    other_disambiguations = {}
    human_disambiguations = {}
    geo_disambiguations = {}
    number_disambiguations = {}
    categories = {}
    links = {}
    given_names = []
    surnames = []
    stubs = []

    new_filename2title = {}

    counter_all = 0

    for i in range(len(lists)):
        if i % num_processes == process_index:
            filename = lists[i]
            with open(filename) as f:
                text = f.read()
                process_list(text,redirects,aliases)

                counter_all += 1
                if counter_all % 1000 == 0:
                    print("Process " + str(process_index) + ', lists processed: ' + str(counter_all), end='\r')

    print("Process " + str(process_index) + ' done with lists.')

    for i in range(len(filenames)):
        if i % num_processes == process_index:
            filename = filenames[i]
            title = filename2title[filename]
            if title in title2Id:
                title_id = title2Id[title]

                with open(filename) as f:
                    text = f.read()
                    process_article(text,
                                    outputpath,
                                    title,
                                    stubs,
                                    categories,
                                    human_disambiguations,
                                    geo_disambiguations,
                                    number_disambiguations,
                                    other_disambiguations,
                                    redirects,
                                    given_names,
                                    surnames,
                                    title_id,
                                    links,
                                    aliases,
                                    new_filename2title)

                    counter_all += 1
                    if counter_all % 1000 == 0:
                        print("Process " + str(process_index) + ', articles processed: ' + str(counter_all), end='\r')



    print("Process " + str(process_index) + ', articles processed: ' + str(counter_all))

    with open(dictionarypath + str(process_index) + '_aliases.json', 'w') as f:
        json.dump(aliases, f)
    with open(dictionarypath + str(process_index) + '_disambiguations_other.json', 'w') as f:
        json.dump(other_disambiguations, f)
    with open(dictionarypath + str(process_index) + '_disambiguations_human.json', 'w') as f:
        json.dump(human_disambiguations, f)
    with open(dictionarypath + str(process_index) + '_disambiguations_geo.json', 'w') as f:
        json.dump(geo_disambiguations, f)
    with open(dictionarypath + str(process_index) + '_disambiguations_number.json', 'w') as f:
        json.dump(number_disambiguations, f)
    with open(dictionarypath + str(process_index) + '_given_names.json', 'w') as f:
        json.dump(given_names, f)
    with open(dictionarypath + str(process_index) + '_surnames.json', 'w') as f:
        json.dump(surnames, f)
    with open(dictionarypath + str(process_index) + '_links.json', 'w') as f:
        json.dump(links, f)
    with open(dictionarypath + str(process_index) + '_stubs.json', 'w') as f:
        json.dump(stubs, f)
    with open(dictionarypath + str(process_index) + '_categories.json', 'w') as f:
        json.dump(categories, f)
    with open(dictionarypath + str(process_index) + '_filename2title_2.json', 'w') as f:
        json.dump(new_filename2title, f)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Process " + str(process_index) + ", elapsed time: %s" % str(datetime.timedelta(seconds=elapsed_time)))


def merge_aliases(partial_aliases, aliases):
    for alias in partial_aliases:
        stripped_alias = alias.strip()
        stripped_alias = stripped_alias[:-2] if stripped_alias.endswith("'s") else stripped_alias
        stripped_alias = stripped_alias[:-1] if stripped_alias.endswith("'") else stripped_alias
        if len(stripped_alias) == 0:
            continue
        if stripped_alias not in aliases:
            aliases[stripped_alias] = {}
        entities = partial_aliases[alias]
        for entity in entities:
            if entity not in aliases[stripped_alias]:
                aliases[stripped_alias][entity] = 0
            aliases[stripped_alias][entity] += partial_aliases[alias][entity]

def merge_links(partial_dict, dict):
    for key in partial_dict:

        if key not in dict:
            dict[key] = {}
        entities = partial_dict[key]
        for entity in entities:
            if entity not in dict[key]:
                dict[key][entity] = 0
            dict[key][entity] += partial_dict[key][entity]

def merge_all_dictionaries(all_titles,title2Id):
    aliases = {}
    other_disambiguations = {}
    human_disambiguations = {}
    geo_disambiguations = {}
    number_disambiguations = {}
    categories = {}
    links = {}
    given_names = []
    surnames = []
    stubs = []

    redirects = json.load(open(dictionarypath + 'redirects.json'))
    filename2title = {}

    for i in range(num_processes):
        partial_aliases = json.load(open(dictionarypath + str(i) + '_aliases.json'))
        merge_aliases(partial_aliases, aliases)

        partial_other_disambiguations = json.load(open(dictionarypath + str(i) + '_disambiguations_other.json'))
        for title in partial_other_disambiguations:
            other_disambiguations[title] = partial_other_disambiguations[title]

        partial_human_disambiguations = json.load(open(dictionarypath + str(i) + '_disambiguations_human.json'))
        for title in partial_human_disambiguations:
            human_disambiguations[title] = partial_human_disambiguations[title]

        partial_geo_disambiguations = json.load(open(dictionarypath + str(i) + '_disambiguations_geo.json'))
        for title in partial_geo_disambiguations:
            geo_disambiguations[title] = partial_geo_disambiguations[title]

        partial_number_disambiguations = json.load(open(dictionarypath + str(i) + '_disambiguations_number.json'))
        for title in partial_number_disambiguations:
            number_disambiguations[title] = partial_number_disambiguations[title]

        partial_given_names = json.load(open(dictionarypath + str(i) + '_given_names.json'))
        given_names.extend(partial_given_names)

        partial_surnames = json.load(open(dictionarypath + str(i) + '_surnames.json'))
        surnames.extend(partial_surnames)

        partial_stubs = json.load(open(dictionarypath + str(i) + '_stubs.json'))
        stubs.extend(partial_stubs)

        partial_categories = json.load(open(dictionarypath + str(i) + '_categories.json'))
        for title in partial_categories:
            categories[title] = partial_categories[title]

        partial_links = json.load(open(dictionarypath + str(i) + '_links.json'))
        merge_links(partial_links, links)

        partial_filename2title = json.load(open(dictionarypath + str(i) + '_filename2title_2.json'))
        for filename in partial_filename2title:
            filename2title[filename] = partial_filename2title[filename]

        '''os.remove(dictionarypath + str(i) + '_aliases.json')
        os.remove(dictionarypath + str(i) + '_disambiguations_other.json')
        os.remove(dictionarypath + str(i) + '_disambiguations_human.json')
        os.remove(dictionarypath + str(i) + '_disambiguations_geo.json')
        os.remove(dictionarypath + str(i) + '_disambiguations_number.json')
        os.remove(dictionarypath + str(i) + '_given_names.json')
        os.remove(dictionarypath + str(i) + '_surnames.json')
        os.remove(dictionarypath + str(i) + '_stubs.json')
        os.remove(dictionarypath + str(i) + '_categories.json')
        os.remove(dictionarypath + str(i) + '_links.json')
        os.remove(dictionarypath + str(i) + '_filename2title_2.json')'''

    with open(dictionarypath + 'aliases.json', 'w') as f:
        json.dump(aliases, f)
    exit(0)
    with open(dictionarypath + 'disambiguations_other.json', 'w') as f:
        json.dump(other_disambiguations, f)
    with open(dictionarypath + 'disambiguations_human.json', 'w') as f:
        json.dump(human_disambiguations, f)
    with open(dictionarypath + 'disambiguations_geo.json', 'w') as f:
        json.dump(geo_disambiguations, f)
    with open(dictionarypath + 'disambiguations_number.json', 'w') as f:
        json.dump(number_disambiguations, f)
    with open(dictionarypath + 'given_names.json', 'w') as f:
        json.dump(given_names, f)
    with open(dictionarypath + 'surnames.json', 'w') as f:
        json.dump(surnames, f)
    with open(dictionarypath + 'links.json', 'w') as f:
        json.dump(links, f)
    with open(dictionarypath + 'stubs.json', 'w') as f:
        json.dump(stubs, f)
    with open(dictionarypath + 'categories.json', 'w') as f:
        json.dump(categories, f)

    given_names = set(given_names)
    surnames = set(surnames)
    stubs = set(stubs)

    entities_not_to_keep = set()
    entities_not_to_keep.update(given_names)
    entities_not_to_keep.update(stubs)
    entities_not_to_keep.update(surnames)
    entities_not_to_keep.update(other_disambiguations.keys())
    entities_not_to_keep.update(geo_disambiguations.keys())
    entities_not_to_keep.update(number_disambiguations.keys())
    entities_not_to_keep.update((human_disambiguations.keys()))

    redirects_reverse = {}
    for redirect in redirects:
        entity = redirects[redirect]
        if len(entity) > 0:
            entity_lower = entity[0].lower() + entity[1:]
            entity_upper = entity[0].upper() + entity[1:]
            if entity_lower in all_titles:
                entity = entity_lower
            elif entity_upper in all_titles:
                entity = entity_upper
            else:
                continue

            if entity not in redirects_reverse:
                redirects_reverse[entity] = []
            redirects_reverse[entity].append(redirect.strip())

    cleaned_aliases = {}
    aliases_reverse = {}

    for alias in aliases:
        if len(alias) <= 1:
            continue

        entities = aliases[alias]
        for original_entity in entities:
            if len(original_entity) == 0 or original_entity in entities_not_to_keep:
                continue
            entity_lower = original_entity[0].lower() + original_entity[1:]
            entity_upper = original_entity[0].upper() + original_entity[1:]
            if entity_lower in all_titles:
                entity = entity_lower
            elif entity_upper in all_titles:
                entity = entity_upper
            else:
                continue

            appearances = entities[original_entity]
            if appearances > 1:
                if entity not in aliases_reverse:
                    aliases_reverse[entity] = {}
                if alias not in aliases_reverse[entity]:
                    aliases_reverse[entity][alias] = 0
                aliases_reverse[entity][alias] += appearances
                if alias not in cleaned_aliases:
                    cleaned_aliases[alias] = {}
                if entity not in cleaned_aliases[alias]:
                    cleaned_aliases[alias][entity] = 0
                cleaned_aliases[alias][entity] += appearances

    aliases_pruned = {}
    aliases_reverse_pruned = {}
    most_popular_entities = []
    redirects_pruned = {}
    redirects_reverse_pruned = {}
    for entity in aliases_reverse:
        aliases_reverse_pruned[entity] = {}
        entity_aliases = aliases_reverse[entity]
        upper = 0
        lower = 0
        for alias in entity_aliases:
            appearances = entity_aliases[alias]
            if alias[0].isupper():
                upper += appearances
            else:
                lower += appearances

            if appearances > 1:
                aliases_reverse_pruned[entity][alias] = appearances

        if upper < lower or len(aliases_reverse_pruned[entity]) == 0:
            #Does not seem to be a named entity.
            del aliases_reverse_pruned[entity]
        else:
            most_popular_entities.append((entity, upper + lower))
            for alias in entity_aliases:
                appearances = entity_aliases[alias]
                if appearances > 1:
                    if alias not in aliases_pruned:
                        aliases_pruned[alias] = {}
                    aliases_pruned[alias][entity] = appearances

            if entity in redirects_reverse:
                redirects_reverse_pruned[entity] = redirects_reverse[entity]

                for redirect in redirects_reverse[entity]:
                    redirects_pruned[redirect] = entity

    most_popular_entities.sort(key=lambda x: x[1], reverse=True)
    most_popular_entities = most_popular_entities[:10000]

    mpe_dict = {}
    for tuple in most_popular_entities:
        mpe_dict[tuple[0]] = tuple[1]


    filename2title_pruned = {}
    for filename in filename2title:
        title = filename2title[filename]
        if title in aliases_reverse_pruned:
            filename2title_pruned[filename] = title

    new_title2Id = {}
    id2title = {}
    for title in title2Id:
        if title in aliases_reverse_pruned:
            new_title2Id[title] = title2Id[title]
            id2title[title2Id[title]] = title

    links_pruned = {}
    for title_id in links:
        if title_id in id2title:
            for entity_id in links[title_id]:
                if entity_id in id2title:
                    if title_id not in links_pruned:
                        links_pruned[title_id] = {}
                    if entity_id not in links_pruned[title_id]:
                        links_pruned[title_id][entity_id] = 0
                    links_pruned[title_id][entity_id] += links[title_id][entity_id]

    with open(dictionarypath + 'links_pruned.json', 'w') as f:
        json.dump(links_pruned, f)
    with open(dictionarypath + 'title2Id_pruned.json', 'w') as f:
        json.dump(new_title2Id, f)
    with open(dictionarypath + 'id2title_pruned.json', 'w') as f:
        json.dump(id2title, f)
    with open(dictionarypath + 'cleaned_aliases.json', 'w') as f:
        json.dump(cleaned_aliases, f)
    with open(dictionarypath + 'aliases_pruned.json', 'w') as f:
        json.dump(aliases_pruned, f)
    with open(dictionarypath + 'aliases_reverse.json', 'w') as f:
        json.dump(aliases_reverse_pruned, f)
    with open(dictionarypath + 'redirects_pruned.json', 'w') as f:
        json.dump(redirects_pruned, f)
    with open(dictionarypath + 'redirects_reverse.json', 'w') as f:
        json.dump(redirects_reverse_pruned, f)
    with open(dictionarypath + 'most_popular_entities.json', 'w') as f:
        json.dump(mpe_dict, f)
    with open(dictionarypath + 'filename2title_2.json', 'w') as f:
        json.dump(filename2title_pruned, f)

if (__name__ == "__main__"):
    config = json.load(open('config/config.json'))
    num_processes = config['processes']
    wikipath = config['wikipath']
    outputpath = config['outputpath']
    dictionarypath = outputpath + 'dictionaries/'
    articlepath = outputpath + ARTICLE_OUTPUTPATH + '/'
    try:
        mode = 0o755
        os.mkdir(articlepath, mode)

        os.mkdir(outputpath + 'disambiguations_human/', mode)
        os.mkdir(outputpath + 'disambiguations_number/', mode)
        os.mkdir(outputpath + 'disambiguations_geo/', mode)
        os.mkdir(outputpath + 'disambiguations_other/', mode)
        os.mkdir(outputpath + 'stubs/', mode)
    except OSError:
        print("directories exist already")

    redirects = json.load(open(dictionarypath + 'redirects.json'))
    title2Id = json.load(open(dictionarypath + 'title2Id.json'))
    filename2title = json.load(open(dictionarypath + 'filename2title_1.json'))
    listof2title = json.load(open(dictionarypath + 'listof2title.json'))
    filenames = list(filename2title.keys())
    lists = list(listof2title.keys())

    print("Read dictionaries.")

    jobs = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=process_articles, args=(i,num_processes, outputpath, redirects, title2Id, filenames, filename2title, lists))
        jobs.append(p)
        p.start()

    del redirects
    del title2Id
    del filename2title
    del filenames
    del lists
    del listof2title

    for job in jobs:
        job.join()

    filename2title = json.load(open(dictionarypath + 'filename2title_1.json'))
    title2Id = json.load(open(dictionarypath + 'title2Id.json'))
    all_titles = set()
    for filename in filename2title:
        all_titles.add(filename2title[filename])

    merge_all_dictionaries(all_titles,title2Id)

