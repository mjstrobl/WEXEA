#!/bin/sh

echo "Start parser 1: Create dictionaries for titles, ids, redirects, categories, lists and store each article separately (including lists and categories)."
python src/parser_1.py

echo "Start parser 2: Separate disambiguation and stub articles; Remove Wikipedia markup; Prune dictionaries."
python src/parser_2.py

echo "Start parser 3: Add new annotations."
python src/parser_3.py

echo "Start parser 4: Co-reference resolution and entity linking."
# python src/parser_4_greedy.py # For non-English Wikipedia dumps.
python src/parser_4.py

echo "Move important files and compress (if desired)."
python src/dataset_creator.py