import re
import os
import time
import json

current_milli_time = lambda: int(round(time.time() * 1000))

IGNORED_NAMESPACES = [
    'wikipedia', 'category', 'file', 'portal', 'template',
    'mediaWiki', 'user', 'help', 'book', 'draft', 'wikiProject',
    'special', 'talk', 'image','module'
]

CONVERT_TEMPLATE_SEPARATORS = {'-':'-','&ndash;':'-','and':' and ','and(-)':' and ','or':' or ','to':' to ','to(-)':' to ','to about':' to about ','+/-':' ± ','±':' ± ','&plusmn;':' ± ','+':' + ',',':', ',', and':', and ',', or':', or ','by':' by ','x':' by ','&times;':' by '}

RE_LINKS = re.compile(r'\[{2}(.*?)\]{2}', re.DOTALL | re.UNICODE)
RE_MENTIONS = re.compile(r'\'\'\'(.*?)\'\'\'', re.DOTALL | re.UNICODE)
RE_EXTERNAL_LINKS = re.compile(r'\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE)
RE_CATEGORIES = re.compile(r'\[\[Category:(.*?)\]\]', re.UNICODE)
RE_TEMPLATE_1 = re.compile(r'{{([^}{]*)}}(;( )?)?', re.DOTALL | re.UNICODE | re.MULTILINE)
RE_CATEGORY_REDIRECT = re.compile(r'{{Category redirect\|([^}{]*)}}', re.DOTALL | re.UNICODE | re.MULTILINE)
RE_REMOVE_SECTIONS = re.compile(r'==\s*See also|==\s*References|==\s*Bibliography|==\s*Sources|==\s*Notes|==\s*Further Reading',re.DOTALL | re.UNICODE)
RE_TABLE = re.compile(r'\{\|(.*?)\|\}', re.DOTALL | re.UNICODE | re.MULTILINE)
RE_FILES = re.compile(r'\[\[(Image|File.*?)\]\]', re.UNICODE)
RE_INFO_OR_TABLE_LINE = re.compile(r'^[\|].*\n?', flags=re.MULTILINE)
RE_COMMENTS = re.compile(r'<!--.*?-->', re.DOTALL | re.UNICODE)
RE_NEWLINES = re.compile(r'\n{2,}', re.DOTALL | re.UNICODE)
RE_FOOTNOTES = re.compile(r'<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE)
RE_MATH = re.compile(r'<math([> ].*?)(</math>|/>)', re.DOTALL | re.UNICODE)
RE_TAGS = re.compile(r'<(.*?)>', re.DOTALL | re.UNICODE)
RE_COMMENTS = re.compile(r'(\n\[\[[a-z][a-z][\w-]*:[^:\]]+\]\])+$', re.UNICODE)
RE_NOWIKI = re.compile(r'<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL | re.UNICODE)
RE_EXTERNAL = re.compile(r'(?<=(\n[ ])|(\n\n)|([ ]{2})|(.\n)|(.\t))(\||\!)([^[\]\n]*?\|)*', re.UNICODE)
RE_BRACKETS = re.compile(r' ?\( *\)')
RE_TABLE_CELL = re.compile(
    r'(\n.{0,4}((bgcolor)|(\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=)|(scope=))(.*))|'
    r'(^.{0,2}((bgcolor)|(\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=))(.*))',
    re.UNICODE
)

RE_ACRONYM = re.compile(r'\s\(([A-Z.]+)\)')

def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_upper(alias):
    for i in range(len(alias)):
        if alias[i].isupper():
            return True
    return False

def intersec(s1, s2):
    s1 = set(s1.split())
    s2 = set(s2.split())
    return len(s1.intersection(s2))

def add_disambgiuation(lines, title, disambiguations, redirects, aliases):
    disambiguations[title] = []
    for line in lines:
        if line.startswith("==See also=="):
            break
        match = re.search(RE_LINKS, line)
        if match:
            entity = match.group(1)
            alias = entity
            pos_bar = entity.find('|')
            if pos_bar > -1:
                alias = entity[pos_bar + 1:]
                entity = entity[:pos_bar]
            if len(entity) > 0:
                entity = add_alias(entity, alias, aliases, redirects)
                disambiguations[title].append(entity)

def add_alias(entity,alias, aliases, redirects):
    if entity.lower().startswith('file:') or entity.lower().startswith('image:') or entity.lower().startswith('list of'):
        return entity

    if entity in redirects:
        entity = redirects[entity]

    if alias not in aliases:
        aliases[alias] = {}
    if entity not in aliases[alias]:
        aliases[alias][entity] = 0
    aliases[alias][entity] += 1

    return entity

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

# Sometimes article entities are mentioned in bold at the beginning of their article, which is captured here.
def find_same_entity_mentions(text, title):
    text = text.strip()
    first_line_break = text.find('\n')
    if first_line_break == -1:
        first_line_break = len(text)
    while True:
        match = re.search(RE_MENTIONS, text[:first_line_break])
        if match:
            mention = match.group(1)
            if ']]' in mention and '[[' in mention:
                text = text[:match.span()[0]] + match.group(1) + text[match.span()[1]:]
            else:
                start = text[:match.span()[0]].rfind('[[')
                if start > -1:
                    before = text[start:match.span()[0]]
                    if ']]' not in before:
                        end = text[match.span()[1]:].find(']]')
                        if end > -1:
                            end += match.span()[1]
                            after = text[match.span()[1]:end]
                            if '[[' not in after:
                                text = text[:match.span()[0]] + mention + text[match.span()[1]:]
                            else:
                                text = text[:match.span()[0]] + '###' + mention + '###' + text[match.span()[1]:]
                        else:
                            text = text[:match.span()[0]] + '###' + mention + '###' + text[match.span()[1]:]
                    else:
                        text = text[:match.span()[0]] + '###' + mention + '###' + text[match.span()[1]:]
                else:
                    text = text[:match.span()[0]] + '###' + mention + '###' + text[match.span()[1]:]
        else:
            break

    text = text.replace("###", "\'\'\'")
    while True:
        match = re.search(RE_MENTIONS, text[:first_line_break])
        if match:
            mention = match.group(1)
            mention = mention.replace('[[', '').replace(']]', '')
            text = text[:match.span()[0]] + '[[' + title + '|' + mention + ']]' + text[match.span()[1]:]
        else:
            break
    return text

def add_link(entity, title_id, title2Id, links):
    if entity in title2Id:
        entity_id = title2Id[entity]
        if entity_id not in links[title_id]:
            links[title_id][entity_id] = 0
        links[title_id][entity_id] += 1

def find_categories(text, categories, title):
    categories[title] = []

    while True:
        match = re.search(RE_CATEGORIES, text)
        if match:
            category = match.group(1)
            pos_bar = category.find('|')
            if pos_bar > -1:
                category = category[:pos_bar]
            categories[title].append(category)
            text = text[:match.span()[0]] + text[match.span()[1]:]
        else:
            break

    return text

def remove_external_links(text):
    external_links = []

    while True:
        match = re.search(RE_EXTERNAL_LINKS, text)
        if match:
            link = match.group(3)
            space = link.find(' ')
            if space > -1:
                external_links.append(link[space:].strip())
            text = text[:match.span()[0]] + text[match.span()[1]:]
        else:
            break

    return text, external_links

def remove_galleries(text):
    while True:
        start = text.lower().find('<gallery')
        if start > -1:
            end = text.find('</gallery>')
            if end > -1:
                while end < start:
                    end += 10
                    end += text[end:].find('</gallery>')

                end += 10
                text = text[:start] + text[end:]
            else:
                break
        else:
            break

    return text

def remove_notes(text):
    text = re.sub(r'^:.*\n?', '', text, flags=re.MULTILINE)
    return text

def process_month(month):
    if "jan" in month or '1' in month:
        return 'January'
    elif "feb" in month or '2' in month:
        return 'February'
    elif "mar" in month or '3' in month:
        return 'March'
    elif "apr" in month or '4' in month:
        return 'April'
    elif "may" in month or '5' in month:
        return 'May'
    elif "jun" in month or '6' in month:
        return 'June'
    elif "jul" in month or '7' in month:
        return 'July'
    elif "aug" in month or '8' in month:
        return 'August'
    elif "sep" in month or '9' in month:
        return 'September'
    elif "oct" in month or '10' in month:
        return 'October'
    elif "nov" in month or '11' in month:
        return 'November'
    elif "dec" in month or '12' in month:
        return 'December'
    else:
        return ''


def process_template(template_lower,template):
    tokens = template.split("|")
    if template_lower.startswith("lang"):
        return tokens[-1]
    elif template_lower.startswith("quote"):
        if len(tokens) == 1:
            return ""
        return '"' + tokens[1] + '"'
    elif template_lower.startswith("wikt-lang") or template_lower.startswith("lang"):
        for i in reversed(range(len(tokens))):
            if '=' not in tokens[i]:
                return tokens[i]
    elif template_lower.startswith('as of'):
        lc = False
        since = False
        alt = None
        df_US = False
        year = None
        month = None
        day = None
        pre = None
        post = None
        if len(tokens) == 1:
            return ""
        else:
            for i in range(1,len(tokens)):
                token = tokens[i]
                if token.strip().startswith('lc') and ('y' in token or ("lc=" in token and len(token) == 3)):
                    lc = True
                elif token.strip().startswith('df'):
                    df_US = True
                elif token.strip().startswith('since') and '=n' not in token and '= n' not in token:
                    since = True
                elif token.strip().startswith("alt"):
                    if '=' in token and len(token.split('=')[1].strip()) > 0:
                        alt = token.split("=")[1]
                elif ("pre=" in token and len(token) > 4) or ("pre =" in token and len(token) > 5):
                    pre = token.split("=")[1]
                elif ("post=" in token and len(token) > 5) or ("post =" in token and len(token) > 6):
                    post = token.split("=")[1]
                elif '=' in token:
                    other_param = token
                elif not year:
                    year = token
                elif not month and len(token) > 0:
                    month = process_month(token.lower())
                elif not day and len(token) > 0:
                    day = token

            if lc:
                if since:
                    replacement = "since "
                else:
                    replacement = "as of "
            else:
                if since:
                    replacement = "Since "
                else:
                    replacement = "As of "

            if alt:
                replacement = alt

            if pre:
                replacement += pre + " "

            if not alt:
                if year and not month and not day:
                    replacement += year
                elif year and month and not day:
                    replacement += month + " " + year
                elif year and month and day:
                    if df_US:
                        replacement += month + " " + day + ", " + year
                    else:
                        replacement += day + " " + month + " " + year

            if post:
                replacement += post + " "

            return replacement
    elif template_lower.startswith('convert') or template_lower.startswith('cvt'):
        if len(tokens) == 1:
            return ""
        else:
            first_value = None
            second_value = None
            separator = None
            unit = None

            for i in range(1, len(tokens)):
                token = tokens[i]
                if not first_value:
                    first_value = token
                elif token in CONVERT_TEMPLATE_SEPARATORS:
                    separator = CONVERT_TEMPLATE_SEPARATORS[token]
                elif separator and not second_value:
                    second_value = token
                else:
                    unit = token
                    break
            if not first_value or len(first_value) == 0 or not unit or len(unit) == 0:
                return ''
            replacement = first_value
            if separator and second_value:
                replacement += separator + second_value

            replacement += ' ' + unit
            return replacement

    return ""

def remove_templates(text):

    text = text.replace('{{spaced ndash}}', ' - ')

    while True:
        m = re.search(RE_TEMPLATE_1, text)
        if m == None:
            break

        start = m.span()[0]
        end = m.span()[1]
        template = m.group(1)
        template_lower = template.lower()
        text_to_keep = process_template(template_lower, template)
        text = text[:start] + text_to_keep + text[end:]

    brackets = []
    idx = 0
    while True:
        start = text[idx:].find('{{')
        if start > -1:
            start += idx
            brackets.append((start, 'opening'))
            idx = start + 2
        else:
            break

    idx = 0
    while True:
        start = text[idx:].find('}}')
        if start > -1:
            start += idx
            brackets.append((start + 2, 'closing'))
            idx = start + 2
        else:
            break

    brackets.sort(key=lambda x: x[0])

    text_to_remove = []
    counter = 0
    start_idx = -1
    end_idx = -1
    for i in range(len(brackets)):
        tuple = brackets[i]
        if tuple[1] == 'opening':
            counter += 1
            if start_idx == -1:
                start_idx = tuple[0]
        else:
            counter -= 1
            end_idx = tuple[0]

        if counter == 0:
            if start_idx > end_idx:
                counter += 1
                continue

            text_to_remove.append((start_idx, end_idx))
            start_idx = -1
            end_idx = -1

    text_to_remove.sort(key=lambda x: x[0], reverse=True)

    offset = 0
    for i in range(len(text_to_remove)):
        tuple = text_to_remove[i]
        removed_text = text[tuple[0] - offset:tuple[1] - offset]
        if removed_text.lower().startswith('{{template'):
            continue
        text = text[:tuple[0] - offset] + text[tuple[1] - offset:]

    return text

def find_positions_of_aliases(previous_text, article_aliases, article_aliases_list, previous_end_index, positions, indices ,aliases_reverse, redirects_reverse, seen_entities, seen_entities_split, entity=None):

    for tuple in article_aliases_list:
        previous_alias = tuple[0]
        alias_regex = tuple[1]

        len_previous_indices = -1
        while True:
            m = re.search(alias_regex, previous_text)
            if m:
                start_previous_alias = m.start(1) + previous_end_index
                positions.append((start_previous_alias, None, previous_text[m.start(1):m.end(1)], "alias_match"))
                previous_text = previous_text[:m.start(1)] + "#" * len(previous_alias) + previous_text[m.end(1):]
                indices.update(set([i for i in range(start_previous_alias, start_previous_alias + len(previous_alias))]))
                if len(indices) == len_previous_indices:
                    break
                len_previous_indices = len(indices)
            else:
                break
    if entity and entity not in seen_entities:
        seen_entities.add(entity)

        for part in entity.split():
            if part[0].isupper():
                if part in seen_entities_split:
                    seen_entities_split[part].append(entity)
                else:
                    seen_entities_split[part] = [entity]

        # add article aliases
        sort = False
        if entity in aliases_reverse:
            for alias in aliases_reverse[entity]:
                if len(alias) == 0 or alias[0].islower() or alias.isnumeric():
                    continue
                appearances = aliases_reverse[entity][alias]
                if alias not in article_aliases:
                    article_aliases[alias] = {}
                    reg = re.escape(alias)
                    article_aliases_list.append((alias, re.compile(rf'\b({reg})\b')))
                    sort = True
                if entity not in article_aliases[alias]:
                    article_aliases[alias][entity] = 0
                article_aliases[alias][entity] += appearances

        if entity in redirects_reverse:
            for redirect in redirects_reverse[entity]:
                if len(redirect) == 0 or redirect[0].islower():
                    continue
                redirect = redirect[:-2] if redirect.endswith("'s") else redirect
                redirect = redirect[:-1] if redirect.endswith("'") else redirect
                if redirect not in article_aliases:
                    article_aliases[redirect] = {}
                    reg = re.escape(redirect)
                    article_aliases_list.append((redirect, re.compile(rf'\b({reg})\b')))
                    sort = True
                if entity not in article_aliases[redirect]:
                    article_aliases[redirect][entity] = 0
                article_aliases[redirect][entity] = 1


        if sort:
            article_aliases_list.sort(key=lambda x: len(x[0]), reverse=True)

def find_acronyms(acronyms, positions, indices, sentence):
    current_start = 0
    while True:
        m = re.search(RE_ACRONYM, sentence[current_start:])
        if m is not None:
            acronym_start = m.span(1)[0] + current_start
            acronym_end = m.span(0)[1] + current_start

            acronym = m.group(1).replace('.', '')

            before = sentence[:m.span(0)[0] + current_start]

            if len(before) == 0 or len(acronym) == 0:
                current_start = acronym_end
                continue

            uppercase_letters = []
            uppercase_letters_string = ''
            for i in range(len(before)):
                c = before[i]
                if c.isupper():
                    uppercase_letters.append(i)
                    uppercase_letters_string += c

            if len(uppercase_letters) >= len(acronym) and uppercase_letters_string[-len(acronym):] == acronym:
                start_idx = uppercase_letters[-len(acronym)]
                actual_entity = before[start_idx:].strip()
                if not actual_entity[-1].isalnum():
                    actual_entity = actual_entity[:-1]

                acronym = m.group(1)

                start = start_idx
                if len(indices.intersection(set([j for j in range(start, start + len(actual_entity))]))) == 0:
                    positions.append((start,None,actual_entity,"acronym_entity"))
                    indices.update(set([i for i in range(start, start + len(actual_entity))]))

                start = acronym_start
                if len(indices.intersection(set([j for j in range(start, start + len(acronym))]))) == 0:
                    positions.append((start, None, acronym, "acronym"))
                    indices.update(set([i for i in range(start, start + len(actual_entity))]))

                    acronyms[acronym] = actual_entity

            current_start = acronym_end
        else:
            break


def find_positions_of_all_links_with_regex(acronyms, text, aliases_reverse, redirects_reverse, redirects, article_aliases, article_aliases_list, seen_entities, seen_entities_split):
    positions = []
    indices = set()
    line_entities = set()

    previous_end_index = 0
    while True:
        match = re.search(RE_LINKS,text[previous_end_index:])
        if match:
            start = match.start() + previous_end_index
            end = match.end() + previous_end_index
            entity = match.group(1)
            alias = entity
            pos_bar = entity.find('|')
            if pos_bar > -1:
                alias = entity[pos_bar + 1:]
                entity = entity[:pos_bar]

            if entity in redirects:
                entity = redirects[entity]

            text = text[:start] + alias + text[end:]
            if entity in aliases_reverse and '#' not in entity:
                positions.append((start,entity,alias,"annotation"))

                previous_text = text[previous_end_index:start]
                find_positions_of_aliases(previous_text, article_aliases, article_aliases_list, previous_end_index, positions, indices, aliases_reverse, redirects_reverse, seen_entities, seen_entities_split, entity=entity)

                previous_end_index = start + len(alias)
                indices.update(set([i for i in range(start, start + len(alias))]))
                line_entities.add(entity)
        else:
            break



    previous_text = text[previous_end_index:]
    find_positions_of_aliases(previous_text, article_aliases, article_aliases_list, previous_end_index, positions,indices, aliases_reverse, redirects_reverse, seen_entities, seen_entities_split)

    find_acronyms(acronyms, positions, indices, text)

    return text, positions, indices, line_entities

def find_entities(text, redirects, aliases, title2Id=None, title_id=-1, links=None):
    idx = 0
    while True:
        match = re.search(RE_LINKS,text[idx:])
        if match:
            entity = match.group(1)
            if '[[' not in entity:
                alias = entity
                pos_bar = entity.find('|')
                if pos_bar > -1:
                    alias = entity[pos_bar + 1:]
                    entity = entity[:pos_bar]
                if len(entity) > 0:
                    entity = add_alias(entity, alias, aliases, redirects)
                    if title_id > -1:
                        add_link(entity, title_id, title2Id, links)

            idx += match.start() + 2
        else:
            break

def remove_irrelevant_sections(text):
    match = re.search(RE_REMOVE_SECTIONS, text)
    if match:
        text = text[:match.span()[0]]
    return text

def remove_tables(text):
    while True:
        match = re.search(RE_TABLE, text)
        if match:
            start = match.span()[0]
            end = match.span()[1]
            table = match.group(1)
            new_open = table.find('{|')
            if new_open > -1:
                text = text[:start + new_open] + text[start + new_open + 4:end - 2] + text[end:]
            else:
                text = text[:start] + text[end:]
        else:
            break

    return text

def remove_files(text):
    while True:
        match = re.search(RE_FILES, text)
        if match:
            start = match.span()[0]
            end = match.span()[1]
            if '[[' in text[start+2:end]:
                # in this case we need to figure out where the annotation ends. The regex cannot figure that out.
                t = text[start:]
                annotation_start = start
                new_line = t.find('\n')
                if new_line == -1:
                    annotation_end = len(text)
                else:
                    annotation_end = new_line + start

                idx = 0
                open_brackets = 0
                close_brackets = 0

                while True:
                    open = t[idx:].find('[[')
                    close = t[idx:].find(']]')

                    if open > -1 and close > -1 and open < close:
                        open_brackets += 1
                        idx += open + 2
                    elif close > -1:
                        close_brackets += 1
                        idx  += close + 2
                    else:
                        break

                    if open_brackets == close_brackets:
                        annotation_end = idx + start
                        break

                    if len(t) <= idx:
                        break

                text = text[:annotation_start] + text[annotation_end:]
            else:
                text = text[:match.span()[0]] + text[match.span()[1]:]
        else:
            break

    return text

def clean_text(text):
    text = text.replace('{{snd}}', ' - ')
    text = text.replace('&ndash;', ' - ')
    text = text.replace('&mdash;', ' - ')
    text = text.replace('&nbsp;', ' ')
    text = re.sub(RE_COMMENTS, '', text)
    text = re.sub(RE_FOOTNOTES, '', text)
    text = re.sub(RE_MATH, '', text)
    text = re.sub(RE_TAGS, '', text)
    text = re.sub(RE_NEWLINES, '\n\n', text).strip()
    text = text.replace('&nbsp;', ' ')
    text = re.sub(RE_INFO_OR_TABLE_LINE, '', text)
    text = re.sub(RE_EXTERNAL, '', text)
    text = re.sub(RE_COMMENTS, '', text)
    text = re.sub(RE_NOWIKI, '', text)
    text = re.sub(RE_TABLE_CELL, '', text)
    text = text.replace("'''", '')
    text = text.replace("''", '"')
    text = text.replace('&thinsp;', '')
    text = re.sub(' +', ' ', text)
    text = re.sub(RE_BRACKETS, '', text)
    return text


if (__name__ == "__main__"):
    config = json.load(open('../config/config.json'))
    num_processes = config['processes']
    wikipath = config['wikipath']
    outputpath = config['outputpath']
    dictionarypath = outputpath + 'dictionaries/'

    redirects = json.load(open(dictionarypath + 'redirects.json'))
    title2Id = json.load(open(dictionarypath + 'title2Id.json'))
    filename2title = json.load(open(dictionarypath + 'filename2title_1.json'))
    filenames = list(filename2title.keys())

    aliases = {}
    title_id = 1
    links = {}
    categories = {}

    counter_all = 0

    for filename in filenames:
        title = filename2title[filename]
        if title not in title2Id:
            continue

        title_id = title2Id[title]
        links[title_id] = {}
        with open(filename) as f:
            text = f.read()

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

        counter_all += 1
        if counter_all % 1000 == 0:
            print('Articles processed: ' + str(counter_all), end='\r')
