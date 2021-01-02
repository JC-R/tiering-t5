import sys
import zipfile
from time import process_time
import numpy as np

import tensorflow as tf
from .cleantext import Cleantext
from .sentence2vec import Sentence2Vec
from .T5Processor import T5Processor


#  initialize the GPU environment
def init_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            #  allow incremental memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


class Embeddings:

    def __init__(self, args):
        # init_gpu()
        self.args = args
        self.segments = []
        self.batch = []
        self.cleaner = Cleantext(args.input, args.max_doc_words, args)
        self.segment_batch_size = int(args.segment_batch_size)
        self.encode_batch_size = args.encode_batch_size
        self.postfix = args.filepostfix
        self.maxlines = args.maxlines
        self.currlines = 0
        self.input_format = args.input_format
        self.t5 = args.t5
        if args.embeddings:
            if self.t5:
                self.vectorizer = T5Processor(args.model, batch_size=args.encode_batch_size)
            else:
                self.vectorizer = Sentence2Vec(args.model, args.device, args.encode_batch_size, args.num_workers)
        self.new_vectors_file()
        if args.output == "-":
            self.cl_zf = sys.stdout
            self.compressed = False
            self.em_zf = None
        else:
            if self.args.cleantext:
                self.new_cleantext_file()

        sys.stderr.write("Starting....\n")

    # using zip format for universal compatibility;
    # change if more compression needed (e.g. if your OS supports it; i.e. 7z)
    def new_cleantext_file(self):
        postfix = ".cleantext.{}".format(self.postfix)
        if self.args.compressed:
            self.cl_zf = zipfile.ZipFile(self.args.output + postfix + ".zip", mode='w',
                                         compression=zipfile.ZIP_DEFLATED)
        else:
            self.cl_zf = open(self.args.output + postfix + ".tsv", "w")

    # using zip format for universal compatibility;
    # change if more compression needed (e.g. if your OS supports it; i.e. 7z)
    def new_vectors_file(self):
        # embeddings always compressed
        postfix = ".embeddings.{}.npz".format(self.postfix)
        self.em_zf = zipfile.ZipFile(self.args.output + "." + self.args.model + postfix, mode='w',
                                     compression=zipfile.ZIP_DEFLATED)

    def dump_numpy_row(self, row):
        tmpname = "{}.npy".format(self.cleaner.documents[self.rowidx])
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
            for doc, body in zip(self.segments, self.batch):
                self.dump_cleantext_row(doc, body)
                self.currlines += 1
            t2 = process_time()
            tt1 = t2 - t1
        if self.args.embeddings:
            t1 = process_time()
            embeddings = self.vectorizer.get_embeddings(self.batch, self.cleaner.doc_processor.doc_groups)
            t2 = process_time()
            tt2 = t2 - t1
            self.dump_embedding(embeddings)
        # advance the file postfix; close and reopen new file
        if 0 < self.maxlines <= self.currlines:
            self.postfix += 1
            if self.args.compressed:
                self.cl_zf.close()
            self.new_cleantext_file()
            self.em_zf.close()
            self.new_vectors_file()
            self.currlines = 0
        return tt1, tt2, process_time() - t0

    def run(self):
        __t0 = process_time()
        start = process_time()
        for idx, (totlines, actual_lines, segment, body, eod) in enumerate(self.cleaner.next_record()):
            self.segments.append(segment)
            self.batch.append(body)
            if eod and len(self.segments) > self.args.segment_batch_size:
                t1, t2, t3 = self.dump()
                self.segments.clear()
                self.batch.clear()
                self.cleaner.clear()
                if self.args.verbose:
                    sys.stderr.write("\r%d (%d, %d): %0.4f, %0.4f, %0.4f" % (totlines,
                                                                 actual_lines,
                                                                 idx,
                                                                 (totlines / __t0),
                                                                 (self.segment_batch_size / t2),
                                                                 (self.segment_batch_size / t3)
                                                                ))
                else:
                    sys.stderr.write("\r%d (%d, %d)" % (totlines, actual_lines, idx))
        if len(self.segments) > 0:
            self.dump()
        end = process_time()
        return (end - start) / 60
