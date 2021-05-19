# WEXEA
Wikipedia EXhaustive Entity Annotator (LREC 2020)

WEXEA is an exhaustive Wikipedia entity annotation system, to create a text corpus based on Wikipedia with exhaustive annotations of entity mentions, i.e. linking all mentions of entities to their corresponding articles. 

This repository is based on our LREC 2020 paper: 

"WEXEA: Wikipedia EXhaustive Entity Annotation"

https://www.aclweb.org/anthology/2020.lrec-1.240

WEXEA runs through several stages of article annotation and the final articles can be found in the 'final_articles' folder in the output directory.
Articles are separately stored in a folder named after the first 3 letters of the title (lowercase) and sentences are split up leading to one sentence per line.
Annotations follow the Wikipedia conventions, just the type of the annotation is added at the end.

This work is still in progress, please email me (mstrobl@ualberta.ca) if you need help or if you have ideas for improvements.

## With basic Entity Linker
This version does not require to download Tensorflow and should be slightly faster. The Entity Linker chooses the candidate with the highest link probability.

1. Install requirements from requirements_basic.txt
2. In config/config.json, provide path of latest wiki dump (xml file) and output path (make sure the output folder does not exist yet, it will be created).
3. Make annotate_basic.sh executable: "chmod 755 annotate_basic.sh"
4. Run annotate_basic.sh with ./annotate_basic.sh


## With neural Entity Linker (as in LREC 2020 paper)

This version uses the neural Entity Linker from https://github.com/nitishgupta/neural-el with small modifications (mainly a predefined set of candidates per entity mention with entities linked in the current article is used and the NER module is removed).


https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96


1. Install from requirements.txt
2. Download the resources folder from https://github.com/nitishgupta/neural-el and set path to models in src/entity_linker/configs/config.ini
3. In config/config.json, provide path of latest wiki dump (xml file) and output path.
4. Since neural-el used models from a tensorflow version with variable names that are not compatible with more recent versions, run src/tf_rename_variables.py with appropriate parameters.
5. Make annotate.sh executable: "chmod 755 annotate.sh"
6. Run annotate.sh with ./annotate.sh

## Visualization

server.py starts a server and opens a website that can be used to visualize an article with Wikipedia links (blue), WEXEA links (red) and unknown entities (green).

## Hardware requirements

32GB of RAM are required (it may work with 16, but not tested) and it should take around 48 hours to finish with a full Wikipedia dump.

## Issues

There are a few issues I'm working on:

- Memory requirement is too high. It does not make sense to keep all dictionaries in memory.
- Sentence break up is not good enough. The nltk sentence tokenizer has to be replaced.
- Our rule-based approach of finding new entities, which cannot be found in Wikipedia can be improved.
- 48 hours is too slow. Once the Wikipedia dump (XML) is parsed, it should be possible to parallelize article processing to speed up the approach significantly.

## Citation (bibtex)

@InProceedings{strobl-trabelsi-zaiane:2020:LREC,
  author    = {Strobl, Michael  and  Trabelsi, Amine  and  Zaiane, Osmar},
  title     = {WEXEA: Wikipedia EXhaustive Entity Annotation},
  booktitle      = {Proceedings of The 12th Language Resources and Evaluation Conference},
  month          = {May},
  year           = {2020},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {1951--1958},
  url       = {https://www.aclweb.org/anthology/2020.lrec-1.240}
}
