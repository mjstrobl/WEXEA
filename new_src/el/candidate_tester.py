import json
import re
from unidecode import unidecode
import os.path
import random
import sqlite3
import torch
import pickle
from tqdm import tqdm
import numpy as np
from model import BertForEntityClassification
from transformers import (
    AdamW, BertTokenizer, BertForNextSentencePrediction
)

RE_LINKS = re.compile(r'\[{2}(.*?)\]{2}', re.DOTALL | re.UNICODE)

MAX_SENT_LENGTH = 128
MAX_ABSTRACT_LENGTH = MAX_SENT_LENGTH / 4

MAX_NUM_CANDIDATES = 10

ABSTRACTS = {}


class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def get_abstract(title, title2filename, entity_title2filename):
    if title in ABSTRACTS:
        return ABSTRACTS[title]
    try:
        with open(title2filename[title].replace("articles_2", "articles_3")) as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    while True:
                        match = re.search(RE_LINKS, line)
                        if match:
                            start = match.start()
                            end = match.end()
                            entity = match.group(1)
                            parts = entity.split('|')
                            entity = parts[-2]

                            line = line[:start] + entity + line[end:]

                        else:
                            break

                    if title in entity_title2filename:
                        with open(entity_title2filename[title]) as fe:
                            entities = fe.read()

                        abstract_parts = line.split(' ')
                        end = int(min(MAX_ABSTRACT_LENGTH, len(abstract_parts)))
                        line = ' '.join(abstract_parts[:end])
                        line += ' ' + entities

                    ABSTRACTS[title] = line
                    return line

    except:
        pass
    return ""


def create_mention_strings(mention):
    mentions = [mention]
    words = mention.split()
    another_mention = ''
    for word in words:
        if len(word) > 1:
            another_mention += word[0].upper() + word[1:].lower() + ' '
        else:
            another_mention += word[0].upper() + ' '

    mentions.append(another_mention.strip())

    mentions.append(mention.lower())

    return mentions


def create_candidate_set(candidates, title2id, type=''):
    current_max_cands = MAX_NUM_CANDIDATES
    #if type == 'train':
    #    current_max_cands = 10

    result = {}
    for key in candidates:
        if key == 'document_mention':
            r = len(candidates[key])
        else:
            r = len(candidates[key]['list'])
            if 'lower' in key:
                r = int(min(current_max_cands, len(candidates[key]['list'])))

            if type == 'train':
                r = int(min(current_max_cands, len(candidates[key]['list'])))

            r = int(min(current_max_cands, len(candidates[key]['list'])))

        for i in range(r):
            if key == 'document_mention':
                name = candidates[key][i][0]
                prior = candidates[key][i][1]
                redirect = candidates[key][i][2]
                surname = candidates[key][i][3]
                document_mention = candidates[key][i][4]

                if name in result:
                    result[name][1] = max(prior, result[name][1])
                    result[name][2] = max(redirect, result[name][2])
                    result[name][3] = max(surname, result[name][3])
                    result[name][4] = max(document_mention, result[name][4])
                else:
                    result[name] = [name, prior, redirect, surname, document_mention]
            else:
                candidate = candidates[key]['list'][i]
                if candidate[0] in title2id:
                    name = candidate[0]
                    prior = candidate[1]
                    redirect = 0
                    surname = 0
                    document_mention = 0
                    if key == 'document_mentions':
                        prior = 0.0
                        document_mention = 1
                    elif prior == -1:
                        redirect = 1
                        prior = 0.0
                    elif prior == -2:
                        surname = 1
                        prior = 0.0

                    if name in result:
                        result[name][1] = max(prior, result[name][1])
                        result[name][2] = max(redirect, result[name][2])
                        result[name][3] = max(surname, result[name][3])
                        result[name][4] = max(document_mention, result[name][4])
                    else:
                        result[name] = [name, prior, redirect, surname, document_mention]

    result_l = []
    for name in result:
        result_l.append((name, result[name][1], result[name][2], result[name][3], result[name][4]))

    return result_l


def get_candidates(mention_upper, id2title, title2id, redirects, person_candidates, priors_lower, document_mentions,
                   type=''):
    mention = unidecode(mention_upper).lower()
    mention_parts = mention_upper.split()
    mention_lower_cleaned = ''
    surname = ''
    for part in mention_parts:
        if len(part) == 2 and part[1] == '.':
            mention_lower_cleaned += ''
        else:
            mention_lower_cleaned += ' ' + unidecode(part).lower()
            surname = part

    if len(surname) > 1:
        surname = surname[0].upper() + surname[1:].lower()

    mention_lower_cleaned = mention_lower_cleaned.strip()

    type_found = []
    candidates = {}

    if mention in priors_lower:
        type_found.append('priors lower')
        candidates['priors_lower'] = priors_lower[mention]

    if mention_lower_cleaned in priors_lower:
        type_found.append('mention_lower_cleaned')
        candidates['mention_lower_cleaned'] = priors_lower[mention_lower_cleaned]

    if "ue" in mention:
        mention = mention.replace('ue', "u")

    if mention not in priors_lower:

        if "oe" in mention:
            mention = mention.replace('oe', "o")

        if mention not in priors_lower:
            if "ae" in mention:
                mention = mention.replace('ae', "a")

    if mention in priors_lower:
        type_found.append('priors lower cleaned')
        candidates['priors_lower_cleaned'] = priors_lower[mention]

    if mention_upper in redirects:
        type_found.append('redirects')
        candidates['redirects'] = {"dict": {redirects[mention_upper]: -1},
                                   'list': [(redirects[mention_upper], -1)]}

    found_in_document_mentions = False
    for document_mention in document_mentions:
        if mention in document_mention:
            type_found.append('document mentions')
            found_in_document_mentions = True
            candidates['document_mention'] = document_mentions[document_mention]
            break

    if not found_in_document_mentions and surname in person_candidates:
        type_found.append('surname')
        candidates['surname'] = person_candidates[surname]

    candidates = create_candidate_set(candidates, title2id, type=type)

    return candidates


def process(documents, entity_start_token_id, id2title, title2id, title2filename, entity_title2filename, redirects,
            person_candidates,
            priors_lower, tokenizer=None, type=''):
    labels = []
    contexts = []
    abstracts = []

    adds_priors = []
    adds_redirects = []
    adds_surname = []

    test_data = {"ids": [], "contexts": [], "candidates": []}

    correct_found = 0
    correct_not_found = 0
    all = 0
    not_found = 0

    counter = {1: 0, 0: 0}

    for document in documents:
        document_mentions = {}
        for i in range(len(document)):
            tuple = document[i]
            sentence_before = []
            sentence_after = []
            if i > 0:
                sentence_before = document[i - 1][0]
            if i < len(document) - 1:
                sentence_after = document[i + 1][0]
            sentence = tuple[0]
            mentions = tuple[1]
            for tuple in mentions:
                all += 1
                start = tuple[0]
                id = tuple[1]

                end = tuple[2]
                mention_upper = ' '.join(sentence[start:end])

                if id in id2title:

                    title = id2title[id]

                    if title in redirects:
                        title = redirects[title]

                    mention = unidecode(' '.join(sentence[start:end]).lower())

                    context = ' '.join(sentence[:start]) + ' <e>' + ' '.join(sentence[start:end]) + '</e> ' + ' '.join(
                        sentence[end:])
                    context = context.strip()

                    if len(sentence_before) + len(sentence) < MAX_SENT_LENGTH / 2:
                        sentence_before_str = ' '.join(sentence_before)
                        # sentence_after_str = ' '.join(sentence_after)
                        context = sentence_before_str + ' ' + context

                        if len(sentence_before) + len(sentence) + len(sentence_after) < MAX_SENT_LENGTH / 2:
                            sentence_after_str = ' '.join(sentence_after)
                            context += ' ' + sentence_after_str

                    candidates = get_candidates(mention_upper, id2title, title2id, redirects, person_candidates,
                                                priors_lower, document_mentions, type=type)

                    if len(candidates) > 0:
                        document_mentions[mention] = candidates
                        candidate_l = []

                        test_data['ids'].append(id)
                        test_data['contexts'].append(context)

                        found = False

                        for j in range(len(candidates)):
                            candidate = candidates[j]
                            if tokenizer != None:
                                abstract = get_abstract(candidate[0], title2filename, entity_title2filename)
                            else:
                                abstract = ''
                            contexts.append(context)
                            abstracts.append(abstract)
                            prior = candidate[1]
                            redirect = candidate[2]
                            surname = candidate[3]

                            adds_priors.append(prior)
                            adds_surname.append(surname)
                            adds_redirects.append(redirect)

                            if candidate[0] == title:
                                labels.append(1)
                                found = True
                            else:
                                labels.append(0)
                            counter[labels[-1]] += 1
                            candidate_l.append((candidate[0], prior, redirect, surname, abstract))

                        if found:
                            correct_found += 1
                        else:
                            correct_not_found += 1

                        test_data['candidates'].append(candidate_l)

                    else:
                        not_found += 1
                        test_data['ids'].append(id)
                        test_data['contexts'].append(context)
                        test_data['candidates'].append([])

                else:
                    not_found += 1
                    test_data['ids'].append(id)
                    test_data['contexts'].append(context)
                    test_data['candidates'].append([])

    dataset = None
    if tokenizer != None:
        inputs = tokenizer(contexts, abstracts, return_tensors='pt', max_length=MAX_SENT_LENGTH, truncation=True,
                           padding='max_length')

        '''input_ids = inputs['input_ids']

        b = []
        for i in range(len(input_ids)):
            # for each mention
            entity_start_token = -1
            for j in range(len(input_ids[i])):
                if input_ids[i][j] == entity_start_token_id:
                    entity_start_token = j
                    break

            entity_mask = [False] * len(input_ids[i])
            entity_mask[entity_start_token] = True
            b.append(entity_mask)

        inputs['entity_mask'] = torch.tensor(b, dtype=torch.bool)'''
        inputs['labels'] = torch.LongTensor([labels]).T
        inputs['priors'] = torch.FloatTensor([adds_priors]).T
        inputs['redirects'] = torch.FloatTensor([adds_redirects]).T
        inputs['surnames'] = torch.FloatTensor([adds_surname]).T
        dataset = OurDataset(inputs)

    print(counter)
    print("all: " + str(all))
    print("correct found: " + str(correct_found))
    print("correct not found: " + str(correct_not_found))
    print("not found: " + str(not_found))

    return dataset, test_data


def get_dataset(wexea_directory, entity_start_token_id, tokenizer=None, type=''):
    fname = "data/" + type + ".pickle"
    if os.path.isfile(fname):
        with open(fname, 'rb') as handle:
            dataset = pickle.load(handle)
            test_data = pickle.load(handle)

        print("using pickled file.")
        return dataset, test_data
    else:

        id2title = json.load(open(wexea_directory + 'dictionaries/id2title.json'))
        title2id = json.load(open(wexea_directory + 'dictionaries/title2Id.json'))
        title2filename = json.load(open(wexea_directory + 'dictionaries/title2filename.json'))
        entity_title2filename = json.load(open(wexea_directory + 'dictionaries/entity_title2filename.json'))
        redirects = json.load(open(wexea_directory + 'dictionaries/redirects.json'))
        person_candidates = json.load(open(wexea_directory + 'dictionaries/person_candidates.json'))
        priors_lower = json.load(open(wexea_directory + 'dictionaries/priors_lower.json'))

        print("get: " + type)
        filename = '../../data/aida_' + type + ".txt"
        documents = []
        with open(filename) as f:
            current_sentence = []
            current_document = []
            mentions = []
            for line in f:
                line = line.strip()
                if line.startswith("DOCSTART") or line.startswith("DOCEND"):
                    if len(current_document) > 0:
                        documents.append(current_document)
                    current_document = []
                    current_sentence = []
                    mentions = []
                elif len(line) == 0 or line == "*NL*":
                    if len(mentions) > 0:
                        current_document.append((current_sentence, mentions))
                    current_sentence = []
                    mentions = []
                else:
                    if line.startswith("MMSTART"):
                        mentions.append([len(current_sentence)])
                        mentions[-1].append(line[8:])
                    elif line.startswith("MMEND"):
                        mentions[-1].append(len(current_sentence))
                    else:
                        current_sentence.append(line)

            if len(current_document) > 0:
                documents.append(current_document)

            #documents = documents[:10]

            dataset, test_data = process(documents, entity_start_token_id, id2title, title2id, title2filename,
                                         entity_title2filename,
                                         redirects, person_candidates, priors_lower, tokenizer=tokenizer, type=type)

            if tokenizer != None:
                with open(fname, 'wb') as handle:
                    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("recreating file.")
            return dataset, test_data


def main():
    config = json.load(open('../../config/config.json'))
    outputpath = config['outputpath']

    wexea_directory = outputpath

    dataset_dev, test_data_dev = get_dataset(wexea_directory, 0 , type='dev')
    dataset_test, test_data_test = get_dataset(wexea_directory, 0, type='test')
    dataset_train, test_data_train = get_dataset(wexea_directory, 0, type='train')


if __name__ == "__main__":
    main()





