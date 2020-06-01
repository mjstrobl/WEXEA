from yattag import Doc
from yattag import indent
import json
import threading
import webbrowser
from http.server import HTTPServer,SimpleHTTPRequestHandler

#ARTICLE_PATH = "/media/michi/Data/wikipedia/coref_articles/articles/"
#ORIGINAL_ARTICLE_PATH = "/media/michi/Data/wikipedia/original_articles/articles/"
TITLE2FILENAME = "/media/michi/Data/wikipedia/coref_articles/articles/title2filename.json"
TITLE2ID = "/media/michi/Data/wikipedia/original_articles/dictionaries/title2Id.json"
WIKILINK = 'https://en.wikipedia.org/?curid='

FILE = 'frontend.html'
PORT = 8080

'''filename2title = json.load(open(FILENAME2TITLE))
title2filename = {}

for filename in filename2title:
    title = filename2title[filename]
    filename = filename.replace(ORIGINAL_ARTICLE_PATH,ARTICLE_PATH)
    title2filename[title] = filename


with open(ARTICLE_PATH + 'title2filename.json' ,'w') as f:
    json.dump(title2filename,f)'''

title2filename = json.load(open(TITLE2FILENAME))
title2id = json.load(open(TITLE2ID))

def process_line(line,doc, tag, text):
    while True:
        start = line.find('[[')
        end = line.find(']]')
        if start > -1 and end > -1 and start < end:
            before = line[:start]
            after = line[end+2:]
            mention = line[start+2:end]

            text(before)

            tokens = mention.split('|')
            entity = tokens[0]
            alias = tokens[1]
            type = tokens[-1]

            if entity in title2id:
                id = title2id[entity]
                link = WIKILINK + str(id)
                klass = 'annotated'
                '''if type == 'COREF':
                    klass = 'coref'
                el'''
                if type == 'RARE_ANNOTATION':
                    klass = 'annotation'
                elif type == 'ANNOTATION':
                    klass = 'annotation'
                '''elif type == 'UNKNOWN':
                    klass = 'unknown'
                elif type == 'REDIRECT':
                    klass = 'annotated'
                '''


                with tag('a',('href',link),('target','_blank'),klass=klass):
                    text(alias)
            else:
                with tag('font', ('color', 'green')):
                    text(alias)

            line = after
        else:
            break

    text(line + " ")

def create_html_paragraph(paragraph,doc, tag, text):
    with tag('p'):
        for line in paragraph:
            process_line(line,doc,tag,text)

def create_html(title2filename,title):

    doc, tag, text = Doc().tagtext()
    print('title: ' + title)
    if title not in title2filename:
        print('title not found.')
        with tag('p'):
            text('Title not available')
    else:
        with tag('h1'):
            text(title)
        with open(title2filename[title]) as f:
            current_paragraph = []
            for line in f:
                line = line.strip()
                if line.startswith('==='):
                    line = line.replace('=','')
                    with tag('h4'):
                        text(line)
                elif line.startswith('=='):
                    line = line.replace('=', '')
                    with tag('h2'):
                        text(line)
                elif len(line) == 0:
                    if len(current_paragraph) > 0:
                        create_html_paragraph(current_paragraph,doc,tag,text)
                    current_paragraph = []
                else:
                    current_paragraph.append(line)

        if len(current_paragraph) > 0:
            create_html_paragraph(current_paragraph,doc,tag,text)

    result = indent(doc.getvalue())
    return result

class TestHandler(SimpleHTTPRequestHandler):

    def do_POST(self):
        """Handle a post request by returning the square of the number."""
        length = int(self.headers.get('content-length'))
        data_string = self.rfile.read(length).decode("utf-8")

        print('data string: ' + data_string)

        html = create_html(title2filename,data_string)

        self.send_response(200)
        self.send_header('Content-Type', 'application/xml')
        self.end_headers()

        self.wfile.write(html.encode())


def open_browser():
    """Start a browser after waiting for half a second."""
    def _open_browser():
        webbrowser.open('http://localhost:%s/%s' % (PORT, FILE))
    thread = threading.Timer(0.5, _open_browser)
    thread.start()

def start_server():
    """Start the server."""
    server_address = ("", PORT)
    server = HTTPServer(server_address, TestHandler)
    server.serve_forever()

if __name__ == "__main__":
    open_browser()
    start_server()

