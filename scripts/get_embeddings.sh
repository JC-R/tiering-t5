# need environment variables:
#  APPDIR=
#  DATADIR
#  OUTDIR


7z e -so $DATADIR/$1.cleantext.json.7z | LC_ALL=C tr -dc '\0-\177' | python $APPDIR/generate-embeddings.py \
--input - \
--max_size 500 \
--model t5-base \
--batch_size 2000 \
--device cuda \
--cleantext \
--embeddings \
--compressed \
--verbose \
--output $OUTDIR/$1 \
--maxlines 500000
$2

# e.g.  $2 -> --lastdoc doc-title    // start AFTER this document


