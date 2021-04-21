#!/bin/sh

echo "start title2Id_redirect_parser"
python src/title2Id_redirect_parser.py
echo "start article_parser_1"
python src/article_parser_1.py
echo "start dicts_creator"
python src/dicts_creator.py
echo "start article_parser_2"
python src/article_parser_2.py
echo "start article_parser_3"
python src/article_parser_3.py
echo "start article_parser_4"
python src/article_parser_4.py
