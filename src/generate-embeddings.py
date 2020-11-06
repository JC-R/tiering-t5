import os
import sys
import json
import re
import zipfile
from time import process_time
import io

import numpy as np
from sentence_transformers import SentenceTransformer
import argparse


class Sentence2Vec():

    # if device=None, let system pick (GPU first)
    def __init__(self, modelpath, device=None, batch_size=32, num_workers=0):
        self.model = SentenceTransformer(modelpath, device=device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch = 0

    def embeddings(self, batch):
        return self.model.encode(batch, self.batch_size, num_workers=self.num_workers)


class Cleantext():

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

    # split the line in segments; enforce max
    def partition(self, body):
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
            docid, url, js = line.split("\t")

            # skip until the start document
            if self.args.lastdoc:
                if not docid == self.args.lastdoc:
                    if self.totlines % 100000 == 0:
                        sys.stderr.write("\rSearching for {} ... lines: {} ".format(self.args.lastdoc, self.totlines))
                    continue
                sys.stderr.write("found at line {} line\n".format(self.totlines))
                self.args.lastdoc = None     # clear the flag
                continue                # start on next document

            body = json.loads(js)["body"]
            body = re.sub(self.remove_chars, ' ', body)
            body = re.sub(self.spaces, ' ', body)
            for idx, section in enumerate(self.partition(body)):
                yield docid + "." + str(idx), section


class Main():

    def __init__(self, args):
        self.args = args
        self.docs = []
        self.batch = []
        self.cleaner = Cleantext(args.input, args.max_doc_size, 20, args)
        self.segment_batch_size = int(args.segment_batch_size)
        self.encode_batch_size = args.encode_batch_size
        self.postfix = args.filepostfix
        self.maxlines = args.maxlines
        self.currlines = 0

        if args.embeddings:
            self.vectorizer = Sentence2Vec(args.model, args.device, args.encode_batch_size, args.num_workers)
        if args.output == "-":
            self.cl_zf = sys.stdout
            self.compressed = False
            self.em_zf = None
        else:
            self.new_cleantext_file()
        self.new_vectors_file()
        sys.stderr.write("Starting....\n")

    # using gz format for universal compatibilty; change if more compression needed
    def new_cleantext_file(self):
        postfix = ".cleantext.{}".format(self.postfix)
        if self.args.compressed:
            self.cl_zf = zipfile.ZipFile(args.output + postfix + ".gz", mode='w', compression=zipfile.ZIP_DEFLATED)
        else:
            self.cl_zf = open(args.output + postfix + ".tsv", "w")

    # using gz format for universal compatibilty; change if more compression needed
    def new_vectors_file(self):
        # embeddings always compressed
        postfix = ".embeddings.{}.npz".format(self.postfix)
        self.em_zf = zipfile.ZipFile(args.output + "." + args.model + postfix, mode='w', compression=zipfile.ZIP_DEFLATED)

    def dump_numpy_row(self, row):
        tmpname = "{}.npy".format(self.docs[self.rowidx])
        self.em_zf.writestr(tmpname, row.tobytes())
        self.rowidx += 1

    def dump_cleantext_row(self, doc, row):
        if self.args.compressed:
            tmpname = "{}.cleantext.tsv".format(doc)
            self.cl_zf.writestr(tmpname, "%s\t%s\n" % (doc, row))
        else:
            self.cl_zf.write("%s\t%s\n" % (doc, row))

    def dump_embedding(self, embeddings):
        if self.em_zf:
            self.rowidx = 0
            np.apply_along_axis(self.dump_numpy_row, axis=1, arr=embeddings)

    def dump(self):
        tt1 = 0
        tt2 = 0
        t0 = process_time()
        if args.cleantext:
            t1 = process_time()
            for doc, body in zip(self.docs, self.batch):
                self.dump_cleantext_row(doc, body)
                self.currlines += 1
            t2 = process_time()
            tt1 = t2-t1
        if self.args.embeddings:
            t1 = process_time()
            embeddings = self.vectorizer.embeddings(self.batch)
            t2 = process_time()
            tt2 = t2-t1
            self.dump_embedding(embeddings)
        # advance the file postfix; close and reopen new file
        if self.currlines >= self.maxlines:
            self.postfix += 1
            if self.args.compressed:
                self.cl_zf.close()
            self.new_cleantext_file()
            self.em_zf.close()
            self.new_vectors_file()
            self.currlines = 0
        return tt1, tt2, process_time()-t0

    def run(self):
        start = process_time()
        for idx, (section, body) in enumerate(self.cleaner.segment()):
            self.docs.append(section)
            self.batch.append(body)
            if idx > 0 and idx % self.args.segment_batch_size == 0:
                t1, t2, t3 = self.dump()
                self.docs.clear()
                self.batch.clear()
                if self.args.verbose:
                    sys.stderr.write("\r%d: %0.4f , %0.4f, %0.4f" % (idx, (t1/self.segment_batch_size), (self.segment_batch_size/t2), t3/self.args.segment_batch_size))
                else:
                    sys.stderr.write("\r%d" % idx)
        if len(self.docs) > 0:
            self.dump()
        end = process_time()
        return (end-start)/60


# call as generate-embeddings.py <input size> <max_source_size> <model>
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-s", "--max_doc_size", type=int, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument("-b", "--segment_batch_size", type=int, required=True)
    parser.add_argument("-c", "--cleantext", action="store_true")
    parser.add_argument("-e", "--embeddings", action="store_true")
    parser.add_argument("--encode_batch_size", type=int, default=32)
    parser.add_argument("-o", "--output", type=str, default="-")
    parser.add_argument("-z", "--compressed", action="store_true", default=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--lastdoc", type=str, help="Start computations AFTER this document")
    parser.add_argument("--filepostfix", type=int, default=0)
    parser.add_argument("--maxlines", type=int, default=1000000)
    parser.add_argument("--num_workers", type=int, default=3)

    args = parser.parse_args()
    
    if args.lastdoc and not args.filepostfix:
        sys.stderr.write("--lastdoc and --filepostfix must be specified together or not at all")
        sys.exit(-1)
    if args.lastdoc:
        sys.stderr.write("Skipping past document {}, file postix {}".format(args.lastdoc, args.filepostfix))

    t = Main(args).run()
    sys.stderr.write("\nDone in %0.2f minutes\n" % t)