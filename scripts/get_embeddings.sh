7z e -so cleantext/$1.cleantext.json.7z | LC_ALL=C tr -dc '\0-\177' | python /home/juan/work/sandbox/tiering-t5/src/main.py --input - \
--max_size 500 --model t5-base --batch_size 2000 --device cuda --cleantext --embeddings \
--compressed --verbose --output $1 --maxlines 500000


