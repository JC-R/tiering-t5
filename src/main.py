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

    def __init__(self, modelpath, device, batch_size):
        self.model = SentenceTransformer(modelpath, device=device)
        self.batch_size = batch_size
        self.batch = 0

    def embeddings(self, batch):
        return self.model.encode(batch)


class Cleantext():

    def __init__(self, inputname, maxsize, padding):
        self.fname = inputname
        # allow some growth
        self.pad = padding
        self.docsize = maxsize - self.pad
        self.bad_content = "bad utf-8 encoding"
        self.remove_chars = r'[\n\|/\x00-\x09\x0c\x0e-0x1f\x80-\xff]'
        self.spaces = r'  +'

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
            if self.bad_content in line:
                continue
            docid, url, js = line.split("\t")
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
        self.cleaner = Cleantext(args.input, args.max_size, 20)
        self.batch_size = int(args.batch_size)
        if args.embeddings:
            self.vectorizer = Sentence2Vec(args.model, args.device, args.max_size)
        if args.output == "-":
            self.cl_zf = sys.stdout
            self.compressed = False
            self.em_zf = None
        else:
            if self.args.compressed:
                self.cl_zf = zipfile.ZipFile(args.output + ".cleantext.gz", mode='w', compression=zipfile.ZIP_DEFLATED)
            else:
                self.cl_zf = open(args.output + ".cleantext.tsv", "w")
            # embeddings always compressed
            self.em_zf = zipfile.ZipFile(args.output + "." + args.model + ".embeddings.npz", mode='w', compression=zipfile.ZIP_DEFLATED)
        sys.stderr.write("Starting....\n")

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
            t2 = process_time()
            tt1 = t2-t1
        if self.args.embeddings:
            t1 = process_time()
            embeddings = self.vectorizer.embeddings(self.batch)
            t2 = process_time()
            tt2 = t2-t1
            self.dump_embedding(embeddings)
        return tt1, tt2, process_time()-t0

    def run(self):
        start = process_time()
        for idx, (section, body) in enumerate(self.cleaner.segment()):
            self.docs.append(section)
            self.bat ch.append(body)
            if idx > 0 and idx % self.args.batch_size == 0:
                t1, t2, t3 = self.dump()
                self.docs.clear()
                self.batch.clear()
                if self.args.verbose:
                    sys.stderr.write("\r%d: %0.4f , %0.4f, %0.4f" % (idx, (t1/self.batch_size), (self.batch_size/t2), t3))
                else:
                    sys.stderr.write("\r%d" % idx)
        if len(self.docs) > 0:
            self.dump()
        end = process_time()
        return (end-start)/60


# call as main.py <input size> <max_source_size> <model>
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-s", "--max_size", type=int, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("-c", "--cleantext", action="store_true")
    parser.add_argument("-e", "--embeddings", action="store_true")
    parser.add_argument("-o", "--output", type=str, default="-")
    parser.add_argument("-z", "--compressed", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    
    t = Main(args).run()
    sys.stderr.write("\nDone in %0.2f minutes\n" % t)
