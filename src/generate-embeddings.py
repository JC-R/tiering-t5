#!/usr/bin/env python

import sys
import argparse

from tiering.embeddings import Embeddings

# call as generate-embeddings.py <input size> <max_source_size> <model>
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str)
parser.add_argument("--t5", action="store_true", help="Use T5 embeddings")
parser.add_argument("-s", "--max_doc_words", type=int, required=True, help="Max # words per sequence")
parser.add_argument("--input", type=str, required=True)
parser.add_argument("-d", "--device", type=str)
parser.add_argument("-b", "--segment_batch_size", type=int, required=True)
parser.add_argument("-c", "--cleantext", action="store_true")
parser.add_argument("-e", "--embeddings", action="store_true")
parser.add_argument("--encode_batch_size", type=int, default=32)
parser.add_argument("-o", "--output", type=str, default="-")
parser.add_argument("-z", "--compressed", action="store_true", default=True)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--maxlines", type=int, default=0, help="Break cleantext output into chunks")
parser.add_argument("--num_workers", type=int, default=3)
parser.add_argument("--input_format", type=str, default="tsv")
parser.add_argument("--is_json", action="store_true")
parser.add_argument("--truncate",  action="store_true")
parser.add_argument("--lastdoc", type=str, help="Start computations AFTER this document")
parser.add_argument("--filepostfix", type=int, default=0)
parser.add_argument("--doclist", type=str, help="Input document filter list; only do docs in this file")

args = parser.parse_args()

if args.lastdoc and not args.filepostfix:
    sys.stderr.write("--lastdoc and --filepostfix must be specified together or not at all")
    sys.exit(-1)
if args.input_format not in ['tsv', 'csv']:
    sys.stderr.write("Unknown input_format. Must be tsv or csv")
    sys.exit(-1)
if args.lastdoc:
    sys.stderr.write("Skipping past document {}, file postix {}".format(args.lastdoc, args.filepostfix))

t = Embeddings(args).run()
sys.stderr.write("\nDone in %0.2f minutes\n" % t)