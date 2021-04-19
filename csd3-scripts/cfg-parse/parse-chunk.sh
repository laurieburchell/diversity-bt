#!/bin/bash

DATA_FOLDER="/home/cs-burc1/projects/diversity-bt/german/parallel/qual75/split-data"
CHUNK=$1

java -mx300g -cp "/home/cs-burc1/apps/stanford-parser-full-2020-11-17/*" edu.stanford.nlp.parser.lexparser.LexicalizedParser -maxLength 80 -nthreads 32 -sentences newline -retainTMPSubcategories -outputFormat 'oneline' edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $DATA_FOLDER/train.en$CHUNK.tail > $DATA_FOLDER/train.en$CHUNK.tail.cfg
