import json
import re
import random
from language_variables import LIST, FILE, IMAGE

months_and_weekdays = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "Noevember", "December",
                       "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}

RE_LINKS = re.compile(r'\[{2}(.*?)\]{2}', re.DOTALL | re.UNICODE)

all_aliases = json.load(open("path/to/dictionaries/aliases_reverse.json"))
all_redirects = json.load(open("path/to/dictionaries/redirects_pruned.json"))
filenames = json.load(open("path/to/dictionaries/filename2title_2.json"))
all_articles = {v: k for k, v in filenames.items()}


def find_entities(text, entities, aliases, entities_to_avoid, replace=False):
    idx = 0
    while True:
        match = re.search(RE_LINKS, text[idx:])
        if match:
            entity = match.group(1)
            article_alias = entity
            parts = entity.split('|')
            if len(parts) > 1:
                article_alias = parts[1]
                entity = parts[0]
            already_replaced = False
            if '[[' not in entity:

                if len(entity) > 0:
                    if entity in all_redirects:
                        entity = all_redirects[entity]
                    if len(article_alias) > 0 and article_alias not in months_and_weekdays and not article_alias.isnumeric() and entity not in entities_to_avoid and entity in all_aliases and not entity[0].islower() and not article_alias[0].islower() and not article_alias[0].isnumeric():
                        if entity not in entities:
                            entities[entity] = []

                        entities[entity].append((idx + match.start(), idx + match.start() + len(article_alias), article_alias))

                        for alias in all_aliases[entity]:
                            if len(alias) > 0 and not alias[0].islower() and not alias in months_and_weekdays and not alias[0].isnumeric():
                                if alias not in aliases:
                                    aliases[alias] = [entity]
                                else:
                                    aliases[alias].append(entity)

                        if replace:
                            text = text[:idx+match.start()] + "#"*len(article_alias) + text[idx+match.end():]
                    else:
                        text = text[:idx + match.start()] + article_alias + text[idx + match.end():]
                        already_replaced = True
                else:
                    text = text[:idx + match.start()] + article_alias + text[idx + match.end():]
                    already_replaced = True


            if not already_replaced and (entity.lower().startswith(FILE + ':') or entity.lower().startswith(
                    IMAGE + ':') or entity.lower().startswith(LIST) or (len(article_alias) > 0 and article_alias[0].islower())):
                if entity in entities:
                    del entities[entity]
                if replace:
                    text = text[:idx + match.start()] + article_alias + text[idx + match.end():]
                else:
                    idx += match.start() + 2
            else:
                idx += match.start() + 2
        else:
            break

    return text

def getLinks(article, entities, aliases, entities_to_avoid):
    if article in all_articles:
        filename = all_articles[article]

        with open(filename) as f:
            for line in f:
                line = line.strip()
                find_entities(line, entities, aliases, entities_to_avoid)

def findAnchors(text):
    anchor_entities = {}
    text = find_entities(text, anchor_entities, {}, {}, replace=True)

    positions = []
    for entity in anchor_entities:
        for pos in anchor_entities[entity]:
            positions.append((pos[0], pos[1], pos[2], entity , "ANCHOR"))

    return text, positions

def removeOwnAnnotations(text, article):
    idx = 0
    while True:
        match = re.search(RE_LINKS, text[idx:])
        if match:
            entity = match.group(1)
            alias = entity
            pos_bar = entity.find('|')
            if pos_bar > -1:
                alias = entity[pos_bar + 1:]
                entity = entity[:pos_bar]

            if entity == article:
                text = text[:idx + match.start()] + alias + text[idx + match.end():]
            else:
                idx += match.start() + 2
        else:
            break

    return text

def cleanWEXEA(text):
    idx = 0
    while True:
        match = re.search(RE_LINKS, text[idx:])
        if match:
            entity = match.group(1)
            parts = entity.split("|")
            if len(parts) == 3:
                alias = parts[0]
                entity = parts[1]
                type = parts[2]
                if "coref" in type.lower() or alias[0].islower() or entity[0].islower() or alias.isnumeric() or alias[0].isnumeric():
                    text_before = text[:idx + match.start()] + parts[1]
                    text = text_before + text[idx + match.end():]
                    idx = len(text_before)
                else:
                    idx += match.end()
            else:
                text_before = text[:idx + match.start()] + parts[0]
                text = text_before + text[idx + match.end():]
                idx = len(text_before)
                #idx += match.end()
        else:
            break

    return text

def checkOverlap(s1, e1, s2, e2):
    x = range(s1, e1)
    y = range(s2, e2)
    xs = set(x)
    if len(xs.intersection(y)) > 0:
        return False
    else:
        return True

def checkAnchorPosition(start, end, anchor_positions):
    for t in anchor_positions:
        if len(t) == 3:
            if not checkOverlap(start, end, t[0], t[1]):
                return False

    return True


def checkEntityPosition(start, end, entity, entity_positions):
    if entity not in entity_positions:
        return True
    else:
        for t in entity_positions[entity]:
            if not checkOverlap(start, end, t[0], t[1]):
                return False

    return True

def findClosestEntity(start, end, entity, anchor_positions, unambiguous_annotations, previous_ambiguous_annotations):
    closest_dist = float("inf")

    for t in anchor_positions:
        if (t[0], t[1], entity) in previous_ambiguous_annotations:
            continue
        if t[3] == entity:
            if t[0] < start:
                if abs(start - t[0]) < closest_dist:
                    closest_dist = abs(start - t[0])
            else:
                if abs(end - t[1]) < closest_dist:
                    closest_dist = abs(end - t[1])

    for t in unambiguous_annotations:
        if t[3] == entity:
            if t[0] < start:
                if abs(start - t[0]) < closest_dist:
                    closest_dist = abs(start - t[0])
            else:
                if abs(end - t[1]) < closest_dist:
                    closest_dist = abs(end - t[1])

    return closest_dist

def findMatches(text, aliases, aliases_sorted, found_positions, anchor_positions):
    found_entity_positions = {}
    for alias in aliases_sorted:
        for m in re.finditer(r'\b%s\b' % re.escape(alias), text):
            start = m.start()
            end = m.end()

            if checkAnchorPosition(start, end, anchor_positions):
                for entity in aliases[alias]:
                    if checkEntityPosition(start, end, entity, found_entity_positions):
                        found_positions.append((start, end, alias, entity))
                        if entity not in found_entity_positions:
                            found_entity_positions[entity] = []
                        found_entity_positions[entity].append((start, end))

def resolveFoundPositions(found_positions, anchor_positions, previous_ambiguous_annotations, type):
    found_positions.sort(key=lambda s: s[0])
    unambiguous_annotations = []
    originally_unambiguous_annotations = []
    ambiguous_annotations = []
    i = 0
    while i < (len(found_positions)):
        okay1 = True
        t1 = found_positions[i]
        if i - 1 >= 0:
            t2 = found_positions[i - 1]
            if checkOverlap(t1[0], t1[1], t2[0], t2[1]):
                okay1 = True
            else:
                okay1 = False

        okay2 = True
        if i + 1 < len(found_positions):
            t2 = found_positions[i + 1]
            if checkOverlap(t1[0], t1[1], t2[0], t2[1]):
                okay2 = True
            else:
                okay2 = False

        if okay1 and okay2:
            unambiguous_annotations.append((t1[0], t1[1], t1[2], t1[3], type))
            originally_unambiguous_annotations.append(found_positions[i])
        else:
            ambiguous_annotations.append(found_positions[i])

        i += 1

    ambiguous_positions = []

    while len(ambiguous_annotations) > 1:
        t1 = ambiguous_annotations.pop(0)
        t2 = ambiguous_annotations.pop(0)

        ambiguous_positions.append(t1)
        ambiguous_positions.append(t2)

        if not checkOverlap(t1[0], t1[1], t2[0], t2[1]):

            if len(t1[2]) > len(t2[2]):
                ambiguous_annotations.insert(0, t1)
            elif len(t2[2]) > len(t1[2]):
                ambiguous_annotations.insert(0, t2)
            else:
                entity1 = t1[3]
                entity2 = t2[3]

                dist1 = findClosestEntity(t1[0], t1[1], entity1, anchor_positions, unambiguous_annotations, previous_ambiguous_annotations)
                dist2 = findClosestEntity(t2[0], t2[1], entity2, anchor_positions, unambiguous_annotations, previous_ambiguous_annotations)
                if dist1 < dist2:
                    ambiguous_annotations.insert(0, t1)
                else:
                    ambiguous_annotations.insert(0, t2)
        else:
            unambiguous_annotations.append((t1[0], t1[1], t1[2], t1[3], type))
            ambiguous_annotations.insert(0, t2)

    if len(ambiguous_annotations) > 0:
        t1 = ambiguous_annotations.pop(0)
        unambiguous_annotations.append((t1[0], t1[1], t1[2], t1[3], type))

    unambiguous_annotations.sort(key=lambda x: x[0], reverse=True)

    return unambiguous_annotations, ambiguous_positions, originally_unambiguous_annotations

def fixFoundPositions(positions, annotations):
    new_positions = []
    for p in positions:
        start_p = p[0]
        end_p = p[1]

        for a in annotations:
            start_a = a[0]
            end_a = a[1]

            if checkOverlap(start_p, end_p, start_a, end_a):
                new_positions.append(p)

    return new_positions

def processArticle(article, filename_winer):

    print(article)
    filename = all_articles[article]
    first_article_aliases = {}
    first_article_entities = {}

    second_article_aliases = {}
    second_article_entities = {}

    third_article_aliases = {}
    third_article_entities = {}

    # Find links in current article
    getLinks(article, first_article_entities, first_article_aliases, {})

    # walk through articles of all of those links and add entities, but without a position.
    for entity in first_article_entities:
        getLinks(entity, third_article_entities, third_article_aliases, first_article_entities)

    for out_link_entity in third_article_entities:
        second_article_entities[out_link_entity] = []
        second_article_aliases[out_link_entity] = [out_link_entity]

    first_article_aliases_sorted = list(first_article_aliases.keys())
    second_article_aliases_sorted = list(second_article_aliases.keys())
    third_article_aliases_sorted = list(third_article_aliases.keys())


    first_article_aliases_sorted.sort(key=lambda s: len(s), reverse=True)
    second_article_aliases_sorted.sort(key=lambda s: len(s), reverse=True)
    third_article_aliases_sorted.sort(key=lambda s: len(s), reverse=True)

    with open(filename) as f:
        original_text = f.read()

        text = removeOwnAnnotations(original_text + "", article)
        text, anchor_positions = findAnchors(text)

        # First rule
        found_positions = []
        findMatches(text, first_article_aliases, first_article_aliases_sorted, found_positions, anchor_positions)
        first_annotations, ambiguous_annotations, first_unambiguous_annotations = resolveFoundPositions(found_positions, anchor_positions, [], "FIRST")

        for t in first_annotations:
            text = text[:t[0]] + "#" * len(t[2]) + text[t[1]:]

        found_positions = []
        already_found_positions = anchor_positions + first_unambiguous_annotations
        findMatches(text, second_article_aliases, second_article_aliases_sorted, found_positions, already_found_positions)
        second_annotations, ambiguous_annotations, second_unambiguous_annotations = resolveFoundPositions(found_positions, already_found_positions, ambiguous_annotations, "SECOND")

        for t in second_annotations:
            text = text[:t[0]] + "#" * len(t[2]) + text[t[1]:]

        # Third rule
        found_positions = []
        already_found_positions = anchor_positions + first_unambiguous_annotations + second_unambiguous_annotations
        findMatches(text, third_article_aliases, third_article_aliases_sorted, found_positions, already_found_positions)
        third_annotations, ambiguous_annotations, unambiguous_annotations = resolveFoundPositions(found_positions, already_found_positions, ambiguous_annotations, "THIRD")
        new_text = text + ""

        current_annotations = anchor_positions + first_annotations + second_annotations + third_annotations
        current_annotations.sort(key=lambda x: x[0], reverse=True)
        for t in current_annotations:
            new_text = new_text[:t[0]] + "[[" + t[3] + "|" + t[2] + "|" + t[4] + "]]" + new_text[t[1]:]

        with open(filename_winer,'w') as f:
            f.write(new_text)

if __name__ == "__main__":
    output_directory_winer = "path/to/wexea_evaluation/winer/"
    output_directory_wexea = "path/to/wexea_evaluation/wexea/"

    sample = random.sample(list(all_articles.keys()), 1000)
    for a in sample:
        filename = all_articles[a].replace("articles_2", "articles_final")

        if not a[0].isalpha():
            continue

        with open(filename) as f:
            original_text = f.read()

            clean_wexea_text = cleanWEXEA(original_text)

            filename = a.replace(" ", '_').replace('/', '_') + '.txt'

            with open(output_directory_wexea + filename, 'w') as f:
                f.write(clean_wexea_text)

        processArticle(a, output_directory_winer + filename)
