#!/bin/bash

# parse target-side CFG trees of Turkish-English data

PARSER="/home/laurie/apps/stanford-parser-full-2020-11-17/*"
DATA_FOLDER="/home/laurie/projects/diversity/turkish-baseline/parallel_data/data/filtered_data"
INPUT="train.en.pp"


java -mx300g -cp "$PARSER" edu.stanford.nlp.parser.lexparser.LexicalizedParser -maxLength 70 -nthreads 36 -sentences newline -retainTMPSubcategories -outputFormat 'oneline' edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $DATA_FOLDER/$INPUT > $DATA_FOLDER/$INPUT.parsed
