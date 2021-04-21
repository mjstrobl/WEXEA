import os
import sys
import json
import pprint
import numpy as np
import tensorflow as tf
import glob
import time

from entity_linker.readers.inference_reader import InferenceReader
from entity_linker.models.figer_model.el_model import ELModel
from entity_linker.readers.config import Config
from entity_linker.readers.vocabloader import VocabLoader

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=7)

pp = pprint.PrettyPrinter()

def process(article,reader,model,output_filename):

    reader.loadTestDoc(article)

    mentions = reader.mentions

    if len(reader.disambiguations) > 0:
        (predTypScNPmat_list,
         widIdxs_list,
         priorProbs_list,
         textProbs_list,
         jointProbs_list,
         evWTs_list,
         pred_TypeSetsList) = model.doInference()

    new_mentions = {}
    mentionnum = 0
    for i in range(len(mentions)):
        mention = mentions[i]
        alias = mention.surface
        mention_start = mention.char_start
        mention_length = mention.link_len
        sent_idx = mention.sentence_idx
        mention_type = mention.type

        if sent_idx not in new_mentions:
            new_mentions[sent_idx] = []

        if i in reader.disambiguations and len(mention.entities) > 1:

            [evWTs, evWIDS, evProbs] = evWTs_list[mentionnum]
            disambiguation = mention.wikititles[evWIDS[2]]

            entity = disambiguation
            mentionnum += 1
        else:
            entity = mention.entities[0]

        new_mentions[sent_idx].append(
            (mention_type, entity, alias, mention_start, mention_length, sent_idx))

    with open(output_filename, 'w') as f:
        sentences = reader.sentences
        for i in range(len(sentences)):
            sentence = sentences[i]
            if i in new_mentions:
                mentions = new_mentions[i]
                mentions.sort(key=lambda x: x[3], reverse=True)

                for mention in mentions:
                    sentence = sentence[:mention[3]] + '[[' + mention[1] + '|' + mention[2] + '|' + mention[
                        0] + ']]' + sentence[mention[3] + mention[4]:]

            f.write(sentence + '\n')


def main():
    config = json.load(open('config/config.json'))

    outputpath = config['outputpath']
    processed_articlepath = outputpath + 'processed_articles/'
    el_articlepath = outputpath + 'neural_el_articles/'
    neural_el_model = config['neural_el_model']

    if not os.path.isdir(el_articlepath):
        try:
            mode = 0o755
            os.mkdir(el_articlepath, mode)
        except FileNotFoundError:
            print('Not found: ' + el_articlepath)
            exit(1)

    config_proto = tf.ConfigProto()
    config_proto.allow_soft_placement = True
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    config = Config('src/entity_linker/configs/config.ini', verbose=False)
    vocabloader = VocabLoader(config)

    reader = InferenceReader(config=config,
                             vocabloader=vocabloader,
                             num_cands=30,
                             batch_size=1,
                             strict_context=False,
                             pretrain_wordembed=True,
                             coherence=True)

    with sess.as_default():
        model = ELModel(
            sess=sess, reader=reader,
            max_steps=32000,
            pretrain_max_steps=32000,
            word_embed_dim=300,
            context_encoded_dim=100,
            context_encoder_num_layers=1,
            context_encoder_lstmsize=100,
            coherence_numlayers=1,
            jointff_numlayers=1,
            learning_rate=0.005,
            dropout_keep_prob=1.0,
            reg_constant=0.00,
            checkpoint_dir="/tmp",
            optimizer='adam',
            strict=False,
            pretrain_word_embed=True,
            typing=True,
            el=True,
            coherence=True,
            textcontext=True,
            WDLength=100,
            Fsize=5,
            entyping=False)

        model.loadModel(ckptpath=neural_el_model)



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
                process(article,reader,model,output_filename)

            end = time.time()
            diff_acc += end - start
            c += len(articles)
            avg = (diff_acc / c) * 1000
            print(str(c) + ', avg t: ' + str(avg), end='\r')

    sys.exit()

if __name__ == '__main__':
    main()
