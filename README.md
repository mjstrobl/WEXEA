# WEXEA

WEXEA is an exhaustive Wikipedia entity annotation system, to create a text corpus based on Wikipedia with exhaustive annotations of entity mentions, i.e. linking all mentions of entities to their corresponding articles.

WEXEA runs through several stages of article annotation and the final articles can be found in the 'final_articles' folder in the output directory.
Articles are separately stored in a folder named after the first 3 letters of the title (lowercase) and sentences are split up leading to one sentence per line.
Annotations follow the Wikipedia conventions, just the type of the annotation is added at the end.

## Downloads

WEXEA for...

1. English: https://drive.google.com/file/d/1xeybIqfctg4nKcwTwibQsZrG1bhqnL4j/view?usp=sharing
2. German: (coming soon)
3. French: (coming soon)
4. Spanish: (coming soon)

## Start CoreNLP toolkit

Download (including models for languages other than English) CoreNLP from https://stanfordnlp.github.io/CoreNLP/index.html

Start server:
```
java -mx16g -cp "<path to corenlp files>" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 -threads 6
```

## Entity Linker

(English only, not relevant for other languages)

Entity Linker including models used from https://github.com/nitishgupta/neural-el. 

Download resources from repository and adjust path to resources folder in ```src/entity_linker/configs/config.ini```.


## Run WEXEA


1. Install requirements from requirements.txt
2. In config/config.json, provide path of latest wiki dump (xml file) and output path (make sure the output folder does not exist yet, it will be created).
3. Make annotate.sh executable: "chmod 755 annotate.sh"
4. Run annotate.sh with ./annotate.sh

## With neural Entity Linker (English only)

Entity Linker including models used from https://github.com/nitishgupta/neural-el. 
Download resources from repository and adjust path to resources folder in src/entity_linker/configs/config.ini.

## Visualization

server.py starts a server and opens a website that can be used to visualize an article with Wikipedia links (blue) and unknown entities (green).

## Hardware requirements

32GB of RAM are required (it may work with 16, but not tested) and it should take around 2-3d to finish with a full English Wikipedia dump (less for other languages).

## Parsers

Time consumption was measured when running on a Ryzen 7 2700X with 64GB of memory. Data was read from and written to a hard drive. Runtimes lower for languages other than English.

### Parser 1 (~2h 45 min / ~4.6GB memory in total / 20,993,369 articles currently):
Create all necessary dictionaries.

### Parser 2 (~1h 45 mins with 6 processes / ~6,000,000 articles to process)
Removes most Wiki markup, irrelevant articles (e.g. lists or stubs), extracts aliases and separates disambiguation pages.

A number of processes can be set to speed up the parsing process of all articles. However, each process consumes around 7.5GB of memory.

### Parser 3 (~2 days with 6 processes / ~2,700,00 articles to process)

Run CoreNLP NER and find other entities based on alias/redirect dictionaries.

### Parser 4 (~2h / ~2,700,00 articles to process)

Run co-reference resolution and EL.

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

