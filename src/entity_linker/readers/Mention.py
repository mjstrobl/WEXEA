"""
Modifications copyright (C) 2022 Michael Strobl
"""

start_word = "<s>"
end_word = "<eos>"
unk_word = "<unk_word>"


class Mention(object):
    def __init__(self, mention_line):
        ''' mention_line : Is the string line stored for each mention
            mid wid wikititle start_token end_token surface tokenized_sentence
            all_types
        '''

        self.start_token = mention_line[0] + 1
        self.end_token = mention_line[1] + 1
        self.surface = mention_line[2]
        self.sent_tokens = [start_word]
        self.sent_tokens.extend(mention_line[3].split(" "))
        self.sent_tokens.append(end_word)
        self.types = ['UNK_TYPES']
        if mention_line[4].strip() == "":
            self.coherence = [unk_word]
        else:
            self.coherence = mention_line[4].split(" ")
        self.wids = mention_line[5]
        self.wikititles = mention_line[6]
        self.entities = mention_line[7]
        self.char_start = mention_line[8]
        self.link_len = mention_line[9]
        self.sentence_idx = mention_line[10]
        self.type = mention_line[11]


    #enddef

    def toString(self):
        outstr = str(self.wikititles) + "\t"
        for i in range(1, len(self.sent_tokens)):
            outstr += self.sent_tokens[i] + " "

        outstr = outstr.strip()
        return outstr

#endclass
