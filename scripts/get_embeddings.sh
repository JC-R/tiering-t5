#!/bin/bash

# may need DATADIR and OUTDIR env vars
# conda activate embeddings

clear

# on my PC, GeForce RTX-1080ti 11Gb, this takes about 9.Gb of gpu ram
#   num_workers = 10
#   encode_batch_size = 300
#   segment batch size -> 3000

# on my RTX2080TI, about 9.7Gb
#   num_workers = 50
#   encode_batch_size = 200
#   segment batch size -> 10000

# process #segment_batch_lines as the multiple of encode_batch and # workers
let w=$3*$2

APPDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

7z e -so ${DATADIR}cleantext/$1.cleantext.json.7z | LC_ALL=C tr -dc '\0-\177' | python $APPDIR/../src/generate-embeddings.py \
--input - \
--model t5-base \
--max_doc_size 500 \
--device cuda \
--cleantext \
--embeddings \
--compressed \
--verbose \
--output ${OUTDIR}$1 \
--maxlines 500000 \
--encode_batch_size $2 \
--num_workers $3 \
--segment_batch_size $w $4 $5


