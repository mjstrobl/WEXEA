import xml.sax
import re
import os
import json
import time
import datetime
from utils import current_milli_time, RE_LINKS, create_file_name_and_directory

from language_variables import RE_CATEGORY_REDIRECT, IGNORED_NAMESPACES, CATEGORY, LIST, REDIRECT

ARTICLE_OUTPUTPATH = "articles_1"

class WikiHandler(xml.sax.ContentHandler):
    def __init__(self, title2Id,redirects,filename2title, categories, category_redirects, category2title, listof2title, list_redirects, outputpath,categorypath, listof_path):
        self.tag = ""
        self.content = ''
        self.title = ''
        self.id = -1
        self.title2Id = title2Id
        self.redirects = redirects
        self.filename2title = filename2title
        self.categories = categories
        self.category_redirects = category_redirects
        self.category2title = category2title
        self.listof2title = listof2title
        self.list_redirects = list_redirects
        self.counter_all = 0
        self.attributes = {}
        self.n = 0
        self.outputpath = outputpath
        self.categorypath = categorypath
        self.listof_path = listof_path
        self.start = time.time()

    # Call when an element starts
    def startElement(self, tag, attributes):
        self.tag = tag
        self.attributes = attributes

    # Call when an elements ends
    def endElement(self, tag):
        if tag == 'title':
            self.title = self.content.strip()
        elif tag == 'id':
            self.id = int(self.content)
            if self.title not in self.title2Id:
                self.title2Id[self.title] = int(self.id)
                self.counter_all += 1

                if self.counter_all % 1000 == 0:
                    diff = time.time() - self.start
                    print('Pages processed: ' + str(self.counter_all) + ', avg t: ' + str(diff / self.counter_all), end='\r')
        elif tag == 'text':
            self.n += 1
            if self.title.lower().startswith(CATEGORY):
                self.processCategory()
            elif self.title.lower().startswith(LIST):
                self.processListOf()
            elif not any(self.title.lower().startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                self.processArticle()
        elif tag == 'redirect' and 'title' in self.attributes:
            redirect = self.attributes['title']
            if redirect.lower().startswith(LIST) or self.title.lower().startswith(LIST):
                self.list_redirects[self.title] = redirect
            elif redirect.lower().startswith(CATEGORY) or self.title.lower().startswith(CATEGORY):
                self.category_redirects[self.title] = redirect
            elif not any(self.title.lower().startswith(ignore + ':') for ignore in IGNORED_NAMESPACES) and not any(redirect.lower().startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                self.redirects[self.title] = redirect

        self.content = ""

    # Call when a character is read
    def characters(self, content):
        self.content += content

    def processCategory(self):
        text = self.content.strip()
        if text.lower().startswith('#' + REDIRECT):
            match = re.search(RE_LINKS,text)
            if match:
                redirect = match.group(1).strip()
                pos_bar = redirect.find('|')
                if pos_bar > -1:
                    redirect = redirect[:pos_bar]
                redirect = redirect.replace('_', ' ')
                self.category_redirects[self.title] = redirect
        else:
            matches = re.findall(RE_CATEGORY_REDIRECT, text)
            if len(matches) == 0:
                self.categories.append(self.title)
                filename = create_file_name_and_directory(self.title[9:], self.categorypath)
                if not filename:
                    return
                try:
                    with open(filename, 'w') as f:
                        f.write(text)

                    self.category2title[filename] = self.title
                except OSError as e:
                    print("Filename too long: %s" % filename)
                    return
            else:
                for match in matches:
                    if not match.lower().startswith(CATEGORY):
                        match = CATEGORY + match
                    self.category_redirects[match] = self.title

    def processListOf(self):
        text = self.content.strip()
        if text.lower().startswith('#' + REDIRECT):
            match = re.search(RE_LINKS,text)
            if match:
                redirect = match.group(1).strip()
                pos_bar = redirect.find('|')
                if pos_bar > -1:
                    redirect = redirect[:pos_bar]
                redirect = redirect.replace('_',' ')
                if not any(redirect.lower().startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                    self.list_redirects[self.title] = redirect
        else:
            filename = create_file_name_and_directory(self.title[8:], self.listof_path)
            if not filename:
                return
            try:
                with open(filename, 'w') as f:
                    f.write(text)

                self.listof2title[filename] = self.title
            except OSError as e:
                print("Filename too long: %s" % filename)
                return

    def processArticle(self):
        text = self.content.strip()
        if text.lower().startswith('#' + REDIRECT):
            match = re.search(RE_LINKS,text)
            if match:
                redirect = match.group(1).strip()
                pos_bar = redirect.find('|')
                if pos_bar > -1:
                    redirect = redirect[:pos_bar]
                redirect = redirect.replace('_', ' ')
                if not any(redirect.lower().startswith(ignore + ':') for ignore in IGNORED_NAMESPACES) and not redirect.lower().startswith(LIST):
                    self.redirects[self.title] = redirect
        else:
            # Store article as separate file

            filename = create_file_name_and_directory(self.title, self.outputpath + ARTICLE_OUTPUTPATH + '/')
            if not filename:
                return

            try:
                with open(filename, 'w') as f:
                    f.write(text)

                self.filename2title[filename] = self.title
            except OSError as e:
                print("Filename too long: %s" % filename)
                return

            # Look for more redirects in article text
            lines = text.split('\n')
            for line in lines:
                if not line.startswith('{{' + REDIRECT):
                    break
                else:
                    line = line[11:]
                    line = line[:line.find('|')]
                    if len(line) > 0:
                        if not any(line.lower().startswith(ignore + ':') for ignore in IGNORED_NAMESPACES) and not line.lower().startswith(LIST):
                            self.redirects[line] = self.title

if (__name__ == "__main__"):
    start_time = time.time()

    title2Id = {}
    redirects = {}
    filename2title = {}
    categories = []
    category_redirects = {}
    category2title = {}
    listof2title = {}
    list_redirects = {}

    config = json.load(open('config/config.json'))

    wikipath = config['wikipath']
    outputpath = config['outputpath']
    dictionarypath = outputpath + 'dictionaries/'
    categorypath = outputpath + 'categories/'
    listof_path = outputpath + 'list_of/'
    articlepath = outputpath + ARTICLE_OUTPUTPATH + '/'

    try:
        mode = 0o755
        #os.mkdir(outputpath, mode)
        os.mkdir(articlepath, mode)
        os.mkdir(categorypath, mode)
        os.mkdir(listof_path, mode)
        os.mkdir(dictionarypath, mode)
    except OSError:
        print("directories exist already")


    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    Handler = WikiHandler(title2Id,redirects,filename2title,categories,category_redirects,category2title, listof2title, list_redirects,outputpath,categorypath,listof_path)
    parser.setContentHandler(Handler)
    parser.parse(wikipath)

    print()
    print("Save dictionaries.")

    id2title = {}
    for title in title2Id:
        id2title[title2Id[title]] = title

    with open(dictionarypath + 'title2Id.json', 'w') as f:
        json.dump(title2Id, f)
    with open(dictionarypath + 'id2title.json', 'w') as f:
        json.dump(id2title, f)
    with open(dictionarypath + 'redirects.json', 'w') as f:
        json.dump(redirects, f)
    with open(dictionarypath + 'filename2title_1.json', 'w') as f:
        json.dump(filename2title, f)
    with open(dictionarypath + 'categories.json', 'w') as f:
        json.dump(categories, f)
    with open(dictionarypath + 'category_redirects.json', 'w') as f:
        json.dump(category_redirects, f)
    with open(dictionarypath + 'category2title.json', 'w') as f:
        json.dump(category2title, f)
    with open(dictionarypath + 'listof2title.json', 'w') as f:
        json.dump(listof2title, f)
    with open(dictionarypath + 'list_redirects.json', 'w') as f:
        json.dump(list_redirects, f)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: %s" % str(datetime.timedelta(seconds=elapsed_time)))