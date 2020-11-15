import sys
import zipfile
from time import process_time
import numpy as np

from .cleantext import Cleantext
from .sentence2vec import Sentence2Vec


class Embeddings:

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
        self.input_format = args.input_format

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

    # using zip format for universal compatibility;
    # change if more compression needed or and your os supports it; i.e. 7z
    def new_cleantext_file(self):
        postfix = ".cleantext.{}".format(self.postfix)
        if self.args.compressed:
            self.cl_zf = zipfile.ZipFile(self.args.output + postfix + ".zip", mode='w', compression=zipfile.ZIP_DEFLATED)
        else:
            self.cl_zf = open(self.args.output + postfix + ".tsv", "w")

    # using zip format for universal compatibility;
    # change if more compression needed or and your os supports it; i.e. 7z
    def new_vectors_file(self):
        # embeddings always compressed
        postfix = ".embeddings.{}.npz".format(self.postfix)
        self.em_zf = zipfile.ZipFile(self.args.output + "." + self.args.model + postfix, mode='w', compression=zipfile.ZIP_DEFLATED)

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
        if self.args.cleantext:
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
        for idx, (totlines, section, body) in enumerate(self.cleaner.segment()):
            self.docs.append(section)
            self.batch.append(body)
            if idx > 0 and idx % self.args.segment_batch_size == 0:
                t1, t2, t3 = self.dump()
                self.docs.clear()
                self.batch.clear()
                if self.args.verbose:
                    sys.stderr.write("\r%d (%d): %0.4f , %0.4f, %0.4f" % (totlines, idx, (t1/self.segment_batch_size),
                                                          (self.segment_batch_size/t2), t3/self.args.segment_batch_size))
                else:
                    sys.stderr.write("\r%d (%d)" % (totlines, idx))
        if len(self.docs) > 0:
            self.dump()
        end = process_time()
        return (end-start)/60


