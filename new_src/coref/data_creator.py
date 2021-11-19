import json
import sys
import copy
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from transformers import AutoConfig,AutoTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import logging
import os
import random
import torch

pronouns = {'I', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themself', 'themselves', 'who', 'whom', 'whose', 'one', "one's",
            'oneself', 'what', 'something', 'anything', 'nothing', 'what', 'which', 'someone', 'anyone', 'no one', 'that', 'this', 'these', 'those', 'former', 'latter',
            'somebody', 'anybody', 'nobody'}

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, subject_mask=None, object_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.subject_mask = subject_mask
        self.object_mask = object_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def create_input_feature(tokens, subject_start_token, object_start_token,label,tokenizer,max_length,
                            cls_token="[CLS]",
                            cls_token_segment_id=0,
                            sep_token_segment_id=1,
                            sep_token="[SEP]",
                            pad_token=0,
                            pad_token_segment_id=0,
                            sequence_a_segment_id=0
                           ):

    token_type_ids = [sequence_a_segment_id] * len(tokens)

    tokens = [cls_token] + tokens + [sep_token]
    token_type_ids = [cls_token_segment_id] + token_type_ids + [sep_token_segment_id]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)

    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
        len(attention_mask), max_length
    )
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
        len(token_type_ids), max_length
    )


    return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, subject_mask=subject_start_token, object_mask=object_start_token, token_type_ids=token_type_ids, label=label)

def create_features(mentions, tokens, tokenizer,max_length, label_counter):
    features = []
    for mention_1 in mentions:
        tuples = mentions[mention_1]
        if len(tuples) > 2:
            for ii in range(len(tuples)):
                for jj in range(len(tuples)):
                    if ii != jj and (tuples[ii][2].lower() in pronouns or tuples[jj][2].lower() in pronouns):
                        if tuples[ii][0] < tuples[jj][0]:
                            current_tokens = []
                            current_tokens.extend(tokens[:tuples[ii][0]])
                            subject_start_token = len(tokens)
                            current_tokens.extend(tokenizer.tokenize("<M1>"))
                            current_tokens.extend(tokens[tuples[ii][0]:tuples[ii][1]])
                            current_tokens.extend(tokenizer.tokenize("</M1>"))
                            current_tokens.extend(tokens[tuples[ii][1]:tuples[jj][0]])
                            object_start_token = len(tokens)
                            current_tokens.extend(tokenizer.tokenize("<M2>"))
                            current_tokens.extend(tokens[tuples[jj][0]:tuples[jj][1]])
                            current_tokens.extend(tokenizer.tokenize("</M2>"))
                            current_tokens.extend(tokens[tuples[jj][1]:])
                            #features.append(create_input_feature(current_tokens, subject_start_token, object_start_token, 1, tokenizer, max_length))
                            label_counter[1] += 1

        for mention_2 in mentions:
            if mention_1 != mention_2:
                for tuple_1 in mentions[mention_1]:
                    for tuple_2 in mentions[mention_2]:
                        positions_1 = set(range(tuple_1[0], tuple_1[1]))
                        positions_2 = set(range(tuple_2[0], tuple_2[1]))
                        if tuple_1[0] < tuple_2[0] and len(positions_1.intersection(positions_2)) == 0 and (tuple_1[2].lower() in pronouns or tuple_2[2].lower() in pronouns):
                            current_tokens = []
                            current_tokens.extend(tokens[:tuple_1[0]])
                            subject_start_token = len(tokens)
                            current_tokens.extend(tokenizer.tokenize("<M1>"))
                            current_tokens.extend(tokens[tuple_1[0]:tuple_1[1]])
                            current_tokens.extend(tokenizer.tokenize("</M1>"))
                            current_tokens.extend(tokens[tuple_1[1]:tuple_2[0]])
                            object_start_token = len(tokens)
                            current_tokens.extend(tokenizer.tokenize("<M2>"))
                            current_tokens.extend(tokens[tuple_2[0]:tuple_2[1]])
                            current_tokens.extend(tokenizer.tokenize("</M2>"))
                            current_tokens.extend(tokens[tuple_2[1]:])
                            #features.append(create_input_feature(current_tokens, subject_start_token,object_start_token, 0, tokenizer,max_length))
                            label_counter[0] += 1

    return features

def create_samples(data_path,prefix,tokenizer,max_length=128):
    filename = data_path + prefix + ".english.v4_gold_conll"
    documents = []
    sentences = []
    sentence = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#end"):
                if len(sentences) > 0:
                    if len(sentence) > 0:
                        sentences.append(sentence)
                    documents.append(sentences)
                    sentences = []
                    sentence = []
                continue
            elif line.startswith("#"):
                continue
            if len(line) == 0 and len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            elif len(line) > 0:
                tokens = line.split()
                word = tokens[3]
                label = tokens[-1]
                sentence.append((word,label))

        if len(sentence) > 0:
            sentence.append((word, label))

        if len(sentences) > 0:
            documents.append(sentences)

    features = []
    label_counter = {0:0,1:0}
    for sentences in documents:
        for i in range(len(sentences)):
            mentions = {}
            tokens = []
            for j in range(i,len(sentences)):
                new_tokens = []
                new_mentions = {}
                mention_start = {}
                for tuple in sentences[j]:
                    word = tuple[0]
                    label = tuple[1]
                    labels = label.split('|')

                    for label in labels:
                        if "(" in label:
                            id = int(label.replace("(", '').replace(")", ''))
                            mention_start[id] = (len(new_tokens) + len(tokens),word)

                    new_tokens.extend(tokenizer.tokenize(word))

                    for label in labels:
                        if ")" in label:
                            id = int(label.replace("(",'').replace(")",''))
                            if id not in new_mentions:
                                new_mentions[id] = []
                            new_mentions[id].append((mention_start[id][0],len(new_tokens) + len(tokens),mention_start[id][1]))

                if len(tokens) + len(new_tokens) <= max_length-6:
                    tokens.extend(new_tokens)
                    for id in new_mentions:
                        if id not in mentions:
                            mentions[id] = new_mentions[id]
                        else:
                            for tuple in new_mentions[id]:
                                mentions[id].append(tuple)
                else:
                    if len(mentions) > 0:
                        # TODO: create features
                        features.extend(create_features(mentions,tokens,tokenizer,max_length,label_counter))
                        tokens = []
                        mentions = {}
                    break

            if len(tokens) > 0 and len(mentions) > 0:
                #TODO: create features
                features.extend(create_features(mentions, tokens, tokenizer, max_length,label_counter))

    return features, label_counter

def main():
    model_name_or_path = "bert-base-cased"
    cache_dir = None
    data_path = "/media/michi/Data/datasets/conll2012/"
    prefix = "train"


    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True
    )

    special_tokens_dict = {'additional_special_tokens': ['<M1>', '</M1>', '<M2>', '</M2>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens')
    #model.resize_token_embeddings(len(tokenizer))

    features, label_counter = create_samples(data_path,prefix,tokenizer)
    print("final: " + str(len(features)))
    print(label_counter)


if (__name__ == "__main__"):
    main()