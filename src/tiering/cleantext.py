import json
import re
import sys


class Cleantext:

    def __init__(self, inputname, max_doc_size, padding, args):
        self.fname = inputname
        # allow some growth
        self.pad = padding
        self.docsize = max_doc_size - self.pad
        self.bad_content = "bad utf-8 encoding"
        self.remove_chars = r'[\n\|/\x00-\x09\x0c\x0e-0x1f\x80-\xff]'
        self.spaces = r'  +'
        self.args = args
        self.totlines = 0
        self.input_format = args.input_format
        self.is_json = args.is_json

    # split the input line into maxsize segments; (enforce max)
    def partition(self, body):
        if self.args.truncate:
            l = min(len(body), self.docsize)
            yield body[0:l].strip()
        else:
            start = 0
            end = self.docsize
            size = len(body)
            while start < size:
                # split at period, space or eol
                while (end - start) < (self.docsize + self.pad) and end < size and not (body[end] in [' ', "."]):
                    end += 1
                yield body[start:end].strip()
                start = end
                end += self.docsize

    def segment(self):
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

            # resuming from a previous run: skip until the start document marker
            if self.args.lastdoc:
                if not docid == self.args.lastdoc:
                    if self.totlines % 100000 == 0:
                        sys.stderr.write("\rSearching for {} ... lines: {} ".format(self.args.lastdoc, self.totlines))
                    continue
                sys.stderr.write("found at line {} line\n".format(self.totlines))
                self.args.lastdoc = None     # clear the flag
                continue                # start on next document

            if self.args.is_json:
                body = json.loads(content)["body"]
                body = re.sub(self.remove_chars, ' ', body)
                body = re.sub(self.spaces, ' ', body)
            else:
                body = content
            for idx, section in enumerate(self.partition(body)):
                yield docid + "." + str(idx), section

