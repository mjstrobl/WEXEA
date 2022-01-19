from stanza.server import CoreNLPClient, StartServer

language = 'fr'

sutime_path = "/home/michi/repos/WEXEA/sutime/"

if language == 'de':
    sutime_rules = "edu/stanford/nlp/models/sutime/defs.sutime.txt,edu/stanford/nlp/models/sutime/english.sutime.txt,edu/stanford/nlp/models/sutime/english.holidays.sutime.txt," + sutime_path + "german.sutime.txt"
    props = {"tokenize.language": "de", "pos.model": "edu/stanford/nlp/models/pos-tagger/german-ud.tagger",
             "tokenize.postProcessor": "edu.stanford.nlp.international.german.process.GermanTokenizerPostProcessor",
             "ner.applyFineGrained": False,
             "ner.model": "edu/stanford/nlp/models/ner/german.distsim.crf.ser.gz",
             "ner.applyNumericClassifiers": True, "ner.useSUTime": True, "ner.language": "fr",
             "sutime.rules": sutime_rules
             }
elif language == 'fr':
    sutime_rules = "edu/stanford/nlp/models/sutime/defs.sutime.txt,edu/stanford/nlp/models/sutime/english.sutime.txt,edu/stanford/nlp/models/sutime/english.holidays.sutime.txt," + sutime_path + "french.sutime.txt"
    props = {"tokenize.language": "fr", "pos.model": "edu/stanford/nlp/models/pos-tagger/french-ud.tagger",
             "ner.applyFineGrained": False,
             "ner.model": "edu/stanford/nlp/models/ner/french-wikiner-4class.crf.ser.gz",
             "ner.applyNumericClassifiers": True, "ner.useSUTime": True, "ner.language": "fr",
             "sutime.rules": sutime_rules
             }

elif language == 'es':
    props = {"tokenize.language": "es", "pos.model": "edu/stanford/nlp/models/pos-tagger/spanish-ud.tagger",
             "ner.applyFineGrained": False,
             "ner.model": "edu/stanford/nlp/models/ner/spanish.ancora.distsim.s512.crf.ser.gz",
             "ner.applyNumericClassifiers": True, "ner.useSUTime": True, "ner.language": "es",
             "sutime.rules": "edu/stanford/nlp/models/sutime/defs.sutime.txt,edu/stanford/nlp/models/sutime/english.sutime.txt,edu/stanford/nlp/models/sutime/english.holidays.sutime.txt,edu/stanford/nlp/models/sutime/spanish.sutime.txt"
             }
else:
    props = {"ner.applyFineGrained": False,
             "ner.model": "edu/stanford/nlp/models/ner/english.muc.7class.distsim.crf.ser.gz"}


#edu/stanford/nlp/models/sutime/defs.sutime.txt,edu/stanford/nlp/models/sutime/english.sutime.txt,edu/stanford/nlp/models/sutime/english.holidays.sutime.txt


annotators = ['tokenize', 'ssplit', 'pos', 'lemma', 'ner']
client = CoreNLPClient(
    annotators=annotators,
    properties=props,
    timeout=60000, endpoint="http://localhost:9000", start_server=StartServer.DONT_START, memory='16g')

line = "Ich habe am Montag den 29.12.1988 Geburtstag."
#line = "My birthday is on Monday."
#line = "Mi cumpleaños es el lunes."
line = "Mon anniversaire est lundi."
annotation = client.annotate(line, properties=props, annotators=annotators)
for i, sent in enumerate(annotation.sentence):
    print(sent)




