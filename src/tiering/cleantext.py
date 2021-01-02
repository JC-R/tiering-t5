import json
import re
import sys

from .document_processor import DocumentProcessor


class Cleantext:

    def __init__(self, inputname, max_tokens, args):
        self.fname = inputname
        self.bad_content = "bad utf-8 encoding"
        self.args = args
        self.totlines = 0
        self.actual_lines = 0
        self.documents = []
        self.input_format = args.input_format
        self.is_json = args.is_json
        self.doc_processor = DocumentProcessor()
        self.max_words = max_tokens
        self.remove_chars = re.compile(r'[\n\|/\x00-\x09\x0c-\x1f\x80-\xff]')
        self.multiple_spaces = re.compile(r'  +')
        self.doclist = None
        if args.doclist:
            sys.stderr.write("Using document filtering....\n")
            self.doclist = set(line.strip() for line in open(args.doclist, 'r'))


    # split an input into maxsize segments; (enforce max # tokens)
    def partition(self, body):
        docs = self.doc_processor.split(body, max_words=self.max_words, truncate=self.args.truncate)
        size = len(docs)
        idx = 0
        for doc in docs:
            idx += 1
            yield doc, idx == size

    # process input from corpus
    #   segment is tsv:   docid \t url \t json-body
    def next_record(self):
        if self.fname == "-":
            # this requires python 3.7+
            # sys.stdin.reconfigure(errors="replace")
            f = sys.stdin
        else:
            f = open(self.fname, "r", encoding="utf8", errors="replace")
        for line in f:
            self.totlines += 1
            if self.bad_content in line:
                continue

            docid, url, content = line.split("\t" if self.input_format == 'tsv' else ',')

            if self.doclist:
                if docid not in self.doclist:
                    continue

            # resuming from a previous run: skip until the start document marker
            if self.args.lastdoc:
                if not docid == self.args.lastdoc:
                    if self.totlines % 100000 == 0:
                        sys.stderr.write("\rSearching for {} ... lines: {} ".format(self.args.lastdoc, self.totlines))
                    continue
                sys.stderr.write("found at line {} line\n".format(self.totlines))
                self.args.lastdoc = None  # clear the flag
                continue  # start on next document

            self.documents.append(docid)

            if self.args.is_json:
                body = json.loads(content)["body"]
                body = self.remove_chars.sub(' ', body)
                body = self.multiple_spaces.sub(' ', body)
            else:
                body = content
            for idx, (section, eod) in enumerate(self.partition(body)):
                yield self.totlines, self.actual_lines, docid + "." + str(idx), section, eod

    def clear(self):
        self.doc_processor.clear()
        self.documents.clear()
