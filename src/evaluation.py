# compare WiNER and WEXEA
import re
import glob
import json

RE_LINKS = re.compile(r'\[{2}(.*?)\]{2}', re.DOTALL | re.UNICODE)
filename2title = json.load(open("path/to/dictionaries/filename2title_3.json"))

def find_entities(text, article):

    idx = 0
    entities = []
    while True:
        match = re.search(RE_LINKS, text[idx:])
        if match:
            original = match.group(1)
            parts = original.split('|')
            if len(parts) > 1:
                alias = parts[1]
                entity = parts[0]
                type = parts[2]
                text_before = text[:idx + match.start()]
                if type != "ANCHOR" and (type != "annotation" or entity == article):
                    entities.append((len(text_before), alias, "[[" + original + "]]"))
                text = text_before + alias + text[idx + match.end():]
                idx += match.start() + len(alias)
            else:
                break
        else:
            break
    entities_cleaned = []
    i = 0
    while i < len(entities)-1:
        entity_1 = entities[i]
        entity_2 = entities[i+1]

        between_text = text[entity_1[0] + len(entity_1[1]):entity_2[0]]
        if len(between_text.strip()) == 0:
            entities_cleaned.append((entity_1[0], text[entity_1[0]:entity_2[0] + len(entity_2[1])], entity_1[2] + " " + entity_2[2]))
            i += 1
        else:
            entities_cleaned.append(entity_1)
        i += 1

    if i < len(entities):
        entities_cleaned.append(entities[i])



    return text, entities_cleaned


d = {}
for filename in filename2title:
    parts = filename.split("/")
    name = parts[-1]
    d[name] = filename2title[filename]


directory = "path/to/winer/*"
files = glob.glob(directory)
n = 0
for file in files:
    filename_winer = file
    filename_wexea = file.replace("winer", "wexea")

    with open(filename_winer) as f:
        content_winer = f.read().strip()

    with open(filename_wexea) as f:
        lines = []
        for line in f:
            lines.append(line)

        new_lines = []
        for i in range(len(lines)):
            line = lines[i]
            if i == 0 or len(lines[i-1].strip()) == 0:
                new_lines.append(line)
            else:
                if len(line.strip()) == 0:
                    new_lines[-1] = new_lines[-1].strip() + line
                else:
                    new_lines[-1] = new_lines[-1].strip() + " " +  line

        content_wexea = '\n'.join(new_lines).strip()

    if "==" in content_winer:
        content_winer = content_winer[:content_winer.find("==")]
    if '==' in content_wexea:
        content_wexea = content_wexea[:content_wexea.find("==")]

    name = file.split("/")[-1]

    if name not in d:
        continue

    content_winer, entities_winer = find_entities(content_winer, d[name])
    content_wexea, entities_wexea = find_entities(content_wexea, d[name])

    if content_wexea != content_winer:
        continue

    n += 1

    n_string = str(n)
    while len(n_string) < 3:
        n_string = '0' + n_string

    with open("path/to/wexea_evaluation/combined/" + n_string + "_" + name, "w") as f:

        current_end_idx = 0
        current_idx_winer = 0
        current_idx_wexea = 0
        while current_idx_winer < len(entities_winer) or current_idx_wexea < len(entities_wexea):
            if current_idx_wexea >= len(entities_wexea):
                start_wexea = len(content_wexea)
                end_wexea = start_wexea
            else:
                entity_wexea = entities_wexea[current_idx_wexea]
                start_wexea = entity_wexea[0]
                end_wexea = start_wexea + len(entity_wexea[1])

            if current_idx_winer >= len(entities_winer):
                start_winer = len(content_winer)
                end_winer = start_winer
            else:
                entity_winer = entities_winer[current_idx_winer]
                start_winer = entity_winer[0]
                end_winer = start_winer + len(entity_winer[1])

            empty_content = content_wexea[current_end_idx:min(start_winer, start_wexea)]+ "\n"
            f.write(empty_content)

            if start_winer < start_wexea:
                if end_winer < start_wexea:
                    f.write(entity_winer[2] + "\t\t\t" + content_wexea[start_winer:end_winer] + "\n")
                    current_idx_winer += 1
                    current_end_idx = end_winer
                else:
                    current_idx_winer += 1
                    current_idx_wexea += 1
                    f.write(entity_winer[2] + " " + content_winer[entity_winer[0] + len(entity_winer[1]):end_wexea] + "\t\t\t" +
                            content_wexea[start_winer:start_wexea] + " " + entity_wexea[2] + "\n")
                    current_end_idx = max(end_winer, end_wexea)
            elif start_winer == start_wexea:
                current_idx_winer += 1
                current_idx_wexea += 1
                current_end_idx = max(end_winer, end_wexea)
                if end_winer < end_wexea:
                    f.write(entity_winer[2] + " " + content_winer[entity_winer[0] + len(entity_winer[1]):end_wexea] + "\t\t\t" +
                        entity_wexea[2] + "\n")
                else:
                    f.write(
                        entity_winer[2] + "\t\t\t" + entity_wexea[2] + " " + content_wexea[end_wexea:end_winer] + "\n")
            else:
                if end_wexea < start_winer:
                    current_idx_wexea += 1
                    f.write(content_winer[start_wexea:end_wexea] + "\t\t\t" + entity_wexea[2] + "\n")
                    current_end_idx = end_wexea
                else:
                    current_idx_winer += 1
                    current_idx_wexea += 1
                    current_end_idx = max(end_winer, end_wexea)
                    f.write(
                         content_winer[start_wexea:start_winer] + " " + entity_winer[2] + "\t\t\t" +
                         entity_wexea[2] + " " + content_wexea[entity_wexea[0] + len(entity_wexea[1]):end_winer] + "\n")

        empty_content = content_wexea[current_end_idx:] + "\n"
        f.write(empty_content)