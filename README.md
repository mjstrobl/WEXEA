# WEXEA
Wikipedia EXhaustive Entity Annotator (LREC 2020)

WEXEA is an exhaustive Wikipedia entity annotation system, to create a text corpus based on Wikipedia with exhaustive annotations of entity mentions, i.e. linking all mentions of entities to their corresponding articles. 

This repository is based on our LREC 2020 paper: 

"WEXEA: Wikipedia EXhaustive Entity Annotation"

https://www.aclweb.org/anthology/2020.lrec-1.240

WEXEA runs through several stages of article annotation and the final articles can be found in the 'final_articles' folder in the output directory.
Articles are separately stored in a folder named after the first 3 letters of the title (lowercase) and sentences are split up leading to one sentence per line.
Annotations follow the Wikipedia conventions, just the type of the annotation is added at the end.

## With basic Entity Linker
This version does not require to download Tensorflow and should be slightly faster. The Entity Linker chooses the candidate with the highest link probability.

1. Install requirements from requirements.txt
2. In config/config.json, provide path of latest wiki dump (xml file) and output path (make sure the output folder does not exist yet, it will be created).
3. Make annotate.sh executable: "chmod 755 annotate.sh"
4. Run annotate.sh with ./annotate.sh


## With neural Entity Linker (as in LREC 2020 paper)

Originally we used a neural entity linker from https://github.com/nitishgupta/neural-el. However, due to compatibility issues with more modern Tensorflow versions, we removed it. Instead the basic entity linker is used by default.


## Visualization

server.py starts a server and opens a website that can be used to visualize an article with Wikipedia links (blue), WEXEA links (red) and unknown entities (green).

## Hardware requirements

32GB of RAM are required (it may work with 16, but not tested) and it should take around 48 hours to finish with a full Wikipedia dump.

## Parsers

Time consumption was measured when running on a Ryzen 7 2700X with 64GB of memory. Data was read from and written to a hard drive.

### Parser 1 (~2h 45 min):
Finds redirect articles as well as redirects mentioned in articles.

Dictionaries:
* title2id: Matching Wiki Ids with their corresponding titles.
* redirects: Stores redirects and their current titles.
* filename2title: Stores filenames, which is used by the next parser.

### Parser 2 (~1h 45 mins with 6 processes / ~6,000,000 articles to process)
Removes most Wiki markup, irrelevant articles (e.g. lists or stubs), extracts aliases and separates disambiguation pages.

Dictionaries:
* aliases: Alias dictionary, storing all names each entity was linked with and stores the number of each appearance.
* disambiguations: List of disambiguation pages (one for geo, human, numbers and others)
* surnames
* given names
* categories: Stores all categories of each article.
* links: Stores the number of times each article appears in each article. IDs instead of titles are used here.
* stubs: Stub articles.

A number of processes can be set to speed up the parsing process of all articles. However, each process consumes around 7.5GB of memory. It is necessary to run `merge_dictionaries.py` afterwards.

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

