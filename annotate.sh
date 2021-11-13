#!/bin/sh

echo "Start parser 1: Create dictionaries for titles, ids, redirects, categories, lists and store each article separately (including lists and categories)."
python new_src/parser_1.py

echo "Start parser 2: Separate disambiguation and stub articles; Remove Wikipedia markup; Prune dictionaries."
python new_src/parser_2.py

echo "Start parser 3: Add new annotations."
python new_src/parser_3.py
