import os
import sys
import json
import pprint
import numpy as np
import tensorflow as tf
import glob
import time

from tensorflow.python import pywrap_tensorflow

from entity_linker.readers.inference_reader import InferenceReader
from entity_linker.models.figer_model.el_model import ELModel
from entity_linker.readers.config import Config
from entity_linker.readers.vocabloader import VocabLoader

from tensorflow.python.training import py_checkpoint_reader


np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=7)

pp = pprint.PrettyPrinter()

def process(article,reader,model,output_filename):

    reader.loadTestDoc(article)


    mentions = reader.mentions


    if reader.disambiguations_counter > 0:
        #print(reader.disambiguations_counter)
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
            print(jointProbs_list)
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
    config = json.load(open('../config/config.json'))

    outputpath = config['outputpath']
    processed_articlepath = outputpath + 'processed_articles/'
    el_articlepath = outputpath + 'neural_el_articles_new/'
    neural_el_model = config['neural_el_model']

    if not os.path.isdir(el_articlepath):
        try:
            mode = 0o755
            os.mkdir(el_articlepath, mode)
        except FileNotFoundError:
            print('Not found: ' + el_articlepath)
            exit(1)

    config_proto = tf.compat.v1.ConfigProto()
    config_proto.allow_soft_placement = True
    config_proto.gpu_options.allow_growth = True
    #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

    tf.compat.v1.disable_eager_execution()





    checkpoint_dir = '/media/michi/Data/non_essential_repos/neural-el/neural-el_resources/models/CDTE.model'
    model_checkpoint_path = '/media/michi/Data/non_essential_repos/neural-el/neural-el_resources/models/CDTE_newstuff222.model'
    replace_from = 'RNN/MultiRNNCell/Cell0/BasicLSTMCell'
    # 'rnn/multirnncell/cell0/basiclstmcell'
    replace_to = 'rnn/multi_rnn_cell/cell_0/basic_lstm_cell'
    add_prefix = None
    state_dict = {}
    with tf.compat.v1.Session() as sess:
        #with sess.as_default():



        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        for var_name, _ in tf.train.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.train.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)
                #new_name = new_name.replace("coherence_layer_0","coherence_layer_0_1")
                new_name = new_name.lower()
                new_name = new_name.replace('linear/matrix', 'kernel')
                new_name = new_name.replace('linear/bias', 'bias')
            if add_prefix:
                new_name = add_prefix + new_name

            if new_name != var_name:
                print('Renaming %s to %s.' % (var_name, new_name))
            var = tf.Variable(var, name=new_name)

        # Save the variables
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, model_checkpoint_path)
        
        '''reader3 = py_checkpoint_reader.NewCheckpointReader(model_checkpoint_path)

        state_dict = {
            v: reader3.get_tensor(v) for v in reader3.get_variable_to_shape_map()
        }'''

        

        '''reader2 = py_checkpoint_reader.NewCheckpointReader(model_checkpoint_path)

        state_dict2 = {
            v: reader2.get_tensor(v) for v in reader2.get_variable_to_shape_map()
        }


        b = 0'''

    #saver = tf.compat.v1.train.Saver()
    #sess=tf.compat.v1.Session()
    #saver.restore(sess, model_checkpoint_path)


    tf.compat.v1.reset_default_graph()


    with tf.compat.v1.Session() as sess:
        config = Config('entity_linker/configs/config.ini', verbose=False)
        vocabloader = VocabLoader(config)
        reader = InferenceReader(config=config,
                                 vocabloader=vocabloader,
                                 num_cands=30,
                                 batch_size=1,
                                 strict_context=False,
                                 pretrain_wordembed=True,
                                 coherence=True)





        for op in tf.compat.v1.get_default_graph().get_operations():
            print(op.name)

        print('####')
        print('####')
        print('####')
        print('####')

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

        model_var_dict = {}
        for op in tf.compat.v1.get_default_graph().get_operations():
            print(op.name)

        model_checkpoint_path = '/media/michi/Data/non_essential_repos/neural-el/neural-el_resources/models/CDTE_newstuff222.model'
        saver = tf.compat.v1.train.Saver()
        #sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess,model_checkpoint_path)




        #model.loadModel(ckptpath=neural_el_model)


        '''model_var_dict = {}
        for op in tf.compat.v1.get_default_graph().get_operations():

            try:
                print(op.name)
                wc_r1 = tf.compat.v1.get_default_graph().get_tensor_by_name(
                    op.name + ":0")
                d = sess.run(wc_r1)
                model_var_dict[op.name] = d
            except:
                print(op.name)'''



        tf.compat.v1.get_default_graph().finalize()

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