#!/bin/bash

#   lastdoc = the last document previously seen; processing will start after this doc
#   filepostfix = the starting segment number (postfix) of the outptu files for this run
#
#
#

# process #segment_batch_lines as the multiple of encode_batch and # workers
let w=$5*$2

# on my PC, GeForce RTX-1080ti 11Gb, this takes about 9Gb of gpu ram
#   num_workers = 10
#   encode_batch_size = 300
#   segment batch size -> 3000


APPDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
$APPDIR/get_embeddings.sh $1 $2 $3 --lastdoc=$4 --filepostfix=$5

#7z e -so cleantext/$1.cleantext.json.7z | LC_ALL=C tr -dc '\0-\177' | python $APPDIR/../src/generate-embeddings.py \
#--input - \
#--model t5-base \
#--max_doc_size 500 \
#--device cuda \
#--cleantext \
#--embeddings \
#--compressed \
#--verbose \
#--output $1 \
#--maxlines 500000 \
#--encode_batch_size $2 \
#--segment_batch_size $w --num_workers $3 --lastdoc $4 --filepostfix $5
