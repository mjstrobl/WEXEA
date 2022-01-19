"""
Modifications copyright (C) 2022 Michael Strobl
"""

import time
import numpy as np
#import utils
#import Mention
from entity_linker.readers import utils
from entity_linker.readers.Mention import Mention
from nltk.tokenize import word_tokenize

start_word = "<s>"
end_word = "<eos>"

class InferenceReader(object):
    def __init__(self, config, vocabloader,
                 num_cands, batch_size, strict_context=True,
                 pretrain_wordembed=True, coherence=True):

        self.typeOfReader = "inference"
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = 'unk'  # In tune with word2vec
        self.unk_wid = "<unk_wid>"
        self.tr_sup = 'tr_sup'
        self.tr_unsup = 'tr_unsup'
        self.pretrain_wordembed = pretrain_wordembed
        self.coherence = coherence
        self.disambiguations = set()

        # Word Vocab
        (self.word2idx, self.idx2word) = vocabloader.getGloveWordVocab()
        self.num_words = len(self.idx2word)

        # Label Vocab
        (self.label2idx, self.idx2label) = vocabloader.getLabelVocab()
        self.num_labels = len(self.idx2label)

        # Known WID Vocab
        (self.knwid2idx, self.idx2knwid) = vocabloader.getKnwnWidVocab()
        self.num_knwn_entities = len(self.idx2knwid)

        # Wid2Wikititle Map
        self.wid2WikiTitle = vocabloader.getWID2Wikititle()
        self.wikiTitle2Wid = {}
        for wid in self.wid2WikiTitle:
            self.wikiTitle2Wid[self.wid2WikiTitle[wid]] = wid

        # Coherence String Vocab
        print("Loading Coherence Strings Dicts ... ")

        (self.cohG92idx, self.idx2cohG9) = utils.load(
            config.cohstringG9_vocab_pkl)
        self.num_cohstr = len(self.idx2cohG9)

        # Crosswikis
        print("Loading Crosswikis dict. (takes ~2 mins to load)")
        self.crosswikis = utils.load(config.crosswikis_pruned_pkl)
        print("Crosswikis loaded. Size: {}".format(len(self.crosswikis)))

        if self.pretrain_wordembed:
            stime = time.time()
            self.word2vec = vocabloader.loadGloveVectors()
            print("[#] Glove Vectors loaded!")
            ttime = (time.time() - stime)/float(60)

        self.batch_size = batch_size
        print("[#] Batch Size: %d" % self.batch_size)
        self.num_cands = num_cands
        self.strict_context = strict_context

        print("\n[#]LOADING COMPLETE")



  #*******************      END __init__      *********************************

    def loadTestDoc(self,article_text):
        #print("[#] Test Mentions File : {}".format(test_mens_file))

        #print("[#] Loading test file and preprocessing ... ")

        self.disambiguations_counter = 0


        self.processTestDoc(article_text)
        self.mention_lines = self.convertSent2NerToMentionLines()
        self.mentions = []
        for line in self.mention_lines:
            m = Mention(line)
            self.mentions.append(m)
            #print(m.toString())

        self.men_idx = 0
        self.num_mens = len(self.mentions)
        self.epochs = 0
        self.disambiguations = set()
        #print("[#] Test Mentions : {}".format(self.num_mens))

    def get_vector(self, word):
        if word in self.word2vec:
            return self.word2vec[word]
        else:
            return self.word2vec['unk']

    def reset_test(self):
        self.men_idx = 0
        self.epochs = 0

    def tokenizeSentence(self,sentence):
        words = []
        mentions = []
        removed = 0
        while True:
            start = sentence.find('[[')
            end = sentence.find(']]')
            if start > -1 and end > -1 and start < end:
                link = sentence[start + 2:end]
                tokens = link.split('|')
                entity = tokens[0]
                alias = entity
                if len(tokens) >= 2:
                    alias = tokens[1]

                words.extend(word_tokenize(sentence[:start]))
                alias_words = word_tokenize(alias)

                entities = entity.split('###')

                mentions.append((len(words), len(alias_words), entities, alias , start+removed, len(link)+4, tokens[-1]))

                words.extend(alias_words)
                removed += end + 2
                sentence = sentence[end + 2:]
            else:
                words.extend(word_tokenize(sentence))
                break

        return words,mentions

    def processTestDoc(self, article_text):
        self.sentidx2ners = {}
        self.sentences_tokenized = []
        self.sentences = []
        idx = 0
        lines = article_text.split('\n')

        for line in lines:
            line = line.strip()
            #if '###' in line:
            self.sentidx2ners[idx] = []
            #print(line)
            words,mentions = self.tokenizeSentence(line)
            #print(line)
            self.sentences_tokenized.append(words)
            self.sentences.append(line)
            for tuple in mentions:
                self.sentidx2ners[idx].append((words, {'start':tuple[0],'end':tuple[0]+tuple[1]-1,'tokens':tuple[3],'entity':tuple[2],'char_start':tuple[4],'link_len':tuple[5],'type':tuple[6]},line))
            idx += 1
            if idx > 3000:
                return

    def convertSent2NerToMentionLines(self):
        '''Convert NERs from document to list of mention strings'''
        mentions = []
        # Make Document Context String for whole document
        cohStr = ""
        for sent_idx, s_nerDicts in self.sentidx2ners.items():
            for s, ner, line in s_nerDicts:
                cohStr += ner['tokens'].replace(' ', '_') + ' '

        cohStr = cohStr.strip()

        for idx in range(0, len(self.sentences_tokenized)):
            if idx in self.sentidx2ners:
                sentence = ' '.join(self.sentences_tokenized[idx])
                s_nerDicts = self.sentidx2ners[idx]
                for tuple in s_nerDicts:
                    s = tuple[0]
                    ner = tuple[1]
                    line = tuple[2]
                    entities = ner['entity']
                    wids = set()
                    wikititles = {}
                    for entity in entities:
                        wid = "unk_wid"
                        original_wid = "unk_wid"
                        wikititle = "unkWT"
                        replaced_entity = entity.replace(' ', '_')
                        if replaced_entity in self.wikiTitle2Wid and self.wikiTitle2Wid[replaced_entity] in self.knwid2idx:
                            original_wid = self.wikiTitle2Wid[replaced_entity]
                            wid = self.knwid2idx[original_wid]
                            wikititle = entity

                            wids.add(wid)
                            wikititles[original_wid] = wikititle
                            wikititles[wid] = wikititle

                    if len(wids) > 1:
                        self.disambiguations_counter += 1

                    mention = (ner['start'],ner['end'],ner['tokens'],sentence,cohStr,wids,wikititles,entities,ner['char_start'],ner['link_len'],idx,ner['type'])
                    '''mention = "%s\t%s\t%s" % ("unk_mid", "unk_wid", "unkWT")
                    mention = mention + str('\t') + str(ner['start'])
                    mention = mention + '\t' + str(ner['end'])
                    mention = mention + '\t' + str(ner['tokens'])
                    mention = mention + '\t' + sentence
                    mention = mention + '\t' + "UNK_TYPES"
                    mention = mention + '\t' + cohStr'''
                    mentions.append(mention)
        return mentions

    def bracketMentionInSentence(self, s, nerDict):
        tokens = s.split(" ")
        start = nerDict['start']
        end = nerDict['end']
        tokens.insert(start, '[[')
        tokens.insert(end + 2, ']]')
        return ' '.join(tokens)

    def _read_mention(self):
        while self.men_idx < len(self.mentions):
            mention = self.mentions[self.men_idx]
            self.men_idx += 1
            candidates = mention.wids
            if len(candidates) > 1:
                self.disambiguations.add(self.men_idx - 1)
                break

        more = 0
        for i in range(self.men_idx,len(self.mentions)):
            candidates = self.mentions[i].wids
            if len(candidates) > 1:
                more += 1
                break

        if more == 0 or self.men_idx == self.num_mens:
            self.men_idx = 0
            self.epochs += 1

        return mention

    def _next_batch(self):
        ''' Data : wikititle \t mid \t wid \t start \t end \t tokens \t labels
        start and end are inclusive
        '''
        # Sentence     = s1 ... m1 ... mN, ... sN.
        # Left Batch   = s1 ... m1 ... mN
        # Right Batch  = sN ... mN ... m1
        (left_batch, right_batch) = ([], [])

        coh_indices = []
        coh_values = []
        if self.coherence:
            coh_matshape = [self.batch_size, self.num_cohstr]
        else:
            coh_matshape = []

        # Candidate WID idxs and their cprobs
        # First element is always true wid
        (wid_idxs_batch, wid_cprobs_batch) = ([], [])

        while len(left_batch) < self.batch_size and self.epochs == 0:
            batch_el = len(left_batch)
            m = self._read_mention()

            # wids : [true_knwn_idx, cand1_idx, cand2_idx, ..., unk_idx]
            # wid_cprobs : [cwikis probs or 0.0 for unks]
            (wid_idxs, wid_cprobs) = self.make_candidates_cprobs(m)
            if len(wid_idxs) == 1:
                wikititle = m.wikititles[wid_idxs[0]]
                m.entities = [wikititle]
            elif len(wid_idxs) == 0:
                m.entities = []
            else:
                while len(wid_idxs) < self.num_cands:
                    wid_idxs.append(0)
                    wid_cprobs.append(0.0)

                # for label in m.types:
                #     if label in self.label2idx:
                #         labelidx = self.label2idx[label]
                #         labels_batch[batch_el][labelidx] = 1.0

                cohFound = False    # If no coherence mention is found, add unk
                if self.coherence:
                    cohidxs = []  # Indexes in the [B, NumCoh] matrix
                    cohvals = []  # 1.0 to indicate presence
                    for cohstr in m.coherence:
                        if cohstr in self.cohG92idx:
                            cohidx = self.cohG92idx[cohstr]
                            cohidxs.append([batch_el, cohidx])
                            cohvals.append(1.0)
                            cohFound = True
                    if cohFound:
                        coh_indices.extend(cohidxs)
                        coh_values.extend(cohvals)
                    else:
                        cohidx = self.cohG92idx[self.unk_word]
                        coh_indices.append([batch_el, cohidx])
                        coh_values.append(1.0)

                # Left and Right context includes mention surface
                left_tokens = m.sent_tokens[0:m.end_token+1]
                right_tokens = m.sent_tokens[m.start_token:][::-1]

                # Strict left and right context
                if self.strict_context:
                    left_tokens = m.sent_tokens[0:m.start_token]
                    right_tokens = m.sent_tokens[m.end_token+1:][::-1]
                # Left and Right context includes mention surface
                else:
                    left_tokens = m.sent_tokens[0:m.end_token+1]
                    right_tokens = m.sent_tokens[m.start_token:][::-1]

                if not self.pretrain_wordembed:
                    left_idxs = [self.convert_word2idx(word)
                                 for word in left_tokens]
                    right_idxs = [self.convert_word2idx(word)
                                  for word in right_tokens]
                else:
                    left_idxs = left_tokens
                    right_idxs = right_tokens

                left_batch.append(left_idxs)
                right_batch.append(right_idxs)


                wid_idxs_batch.append(wid_idxs)
                wid_cprobs_batch.append(wid_cprobs)

        coherence_batch = (coh_indices, coh_values, coh_matshape)

        if len(left_batch) == 0:
            return None
        else:
            return (left_batch, right_batch,
                coherence_batch, wid_idxs_batch, wid_cprobs_batch)

    def print_test_batch(self, mention, wid_idxs, wid_cprobs):
        print("Surface : {}  WID : {}  WT: {}".format(
            mention.surface, mention.wid, self.wid2WikiTitle[mention.wid]))
        print(mention.wid in self.knwid2idx)
        for (idx,cprob) in zip(wid_idxs, wid_cprobs):
            print("({} : {:0.5f})".format(
                self.wid2WikiTitle[self.idx2knwid[idx]], cprob), end=" ")
            print("\n")

    def make_candidates_cprobs(self, m):
        # Fill num_cands now
        surface = utils._getLnrm(m.surface)
        wid_idxs = []
        wid_cprobs = []
        if surface in self.crosswikis:
            # Pruned crosswikis has only known wids and 30 cands at max
            #candwids_cprobs = self.crosswikis[surface][0:self.num_cands-1]
            candwids_cprobs = self.crosswikis[surface]
            (wids, wid_cprobs) = candwids_cprobs
            wid_idxs = [self.knwid2idx[wid] for wid in wids]
            if len(wid_cprobs) > len(wid_idxs):
                wid_cprobs = wid_cprobs[:len(wid_idxs)]

        # All possible candidates added now. Pad with unks
        assert len(wid_idxs) == len(wid_cprobs)
        remain = self.num_cands - len(wid_idxs)
        wid_idxs.extend([0]*remain)
        wid_cprobs.extend([0.0]*remain)

        new_wid_idxs = []
        new_wid_cprobs = []
        for i in range(len(wid_idxs)):
            if wid_idxs[i] in m.wids:
                new_wid_idxs.append(wid_idxs[i])
                new_wid_cprobs.append(wid_cprobs[i])

        if len(new_wid_idxs) == 0:
            for wid in m.wids:
                if wid != 'unk_wid':
                    new_wid_idxs.append(wid)
                    new_wid_cprobs.append(1/len(m.wids))

        return (new_wid_idxs, new_wid_cprobs)

    def embed_batch(self, batch):
        ''' Input is a padded batch of left or right contexts containing words
            Dimensions should be [B, padded_length]
        Output:
            Embed the word idxs using pretrain word embedding
        '''
        output_batch = []
        for sent in batch:
            word_embeddings = [self.get_vector(word) for word in sent]
            output_batch.append(word_embeddings)
        return output_batch

    def embed_mentions_batch(self, mentions_batch):
        ''' Input is batch of mention tokens as a list of list of tokens.
        Output: For each mention, average word embeddings '''
        embedded_mentions_batch = []
        for m_tokens in mentions_batch:
            outvec = np.zeros(300, dtype=float)
            for word in m_tokens:
                outvec += self.get_vector(word)
                outvec = outvec / len(m_tokens)
                embedded_mentions_batch.append(outvec)
        return embedded_mentions_batch

    def pad_batch(self, batch):
        if not self.pretrain_wordembed:
            pad_unit = self.word2idx[self.unk_word]
        else:
            pad_unit = self.unk_word

        lengths = [len(i) for i in batch]
        max_length = max(lengths)
        for i in range(0, len(batch)):
            batch[i].extend([pad_unit]*(max_length - lengths[i]))
        return (batch, lengths)

    def _next_padded_batch(self):
        result = self._next_batch()

        if result != None:
            (left_batch, right_batch,
             coherence_batch,
             wid_idxs_batch, wid_cprobs_batch) = result

            (left_batch, left_lengths) = self.pad_batch(left_batch)
            (right_batch, right_lengths) = self.pad_batch(right_batch)

            if self.pretrain_wordembed:
                left_batch = self.embed_batch(left_batch)
                right_batch = self.embed_batch(right_batch)

            return (left_batch, left_lengths, right_batch, right_lengths,
                    coherence_batch, wid_idxs_batch, wid_cprobs_batch)
        else:
            return None

    def convert_word2idx(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[self.unk_word]

    def next_test_batch(self):
        return self._next_padded_batch()

    def widIdx2WikiTitle(self, widIdx):
        wid = self.idx2knwid[widIdx]
        wikiTitle = self.wid2WikiTitle[wid]
        return wikiTitle
