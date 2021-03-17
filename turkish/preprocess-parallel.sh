#!/bin/bash

moses_scripts="/home/laurie/apps/mosesdecoder/scripts"
entr_path="/mnt/startiger0/laurie/diversity/turkish-baseline/parallel_data"
clean_parallel="/mnt/startiger0/laurie/diversity/turkish-baseline/clean_parallel.py"
src="en"
trg="tr"

mkdir -p $entr_path/data
mkdir -p $entr_path/model

echo "Normalizing punctuation..."

for split in train dev test
do
    for lang in $src $trg
    do
        $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang < $entr_path/data/$split.$lang > $entr_path/data/$split.$lang.norm
    done
done

echo "Truecasing..."

# learn truecasers
for lang in $src $trg
do
    $moses_scripts/recaser/train-truecaser.perl -model $entr_path/model/truecase-model.$lang -corpus $entr_path/data/train.$lang.norm
done

# apply truecaser
for split in train dev test
do
    for lang in $src $trg
    do
        $moses_scripts/recaser/truecase.perl < $entr_path/data/$split.$lang.norm > $entr_path/data/$split.$lang.pp -model $entr_path/model/truecase-model.$lang
    done
done

rm $entr_path/data/*.norm  # remove normed files

echo "filtering bad pairs"
python3 $clean_parallel -src_file $entr_path/data/train.$src.pp -trg_file $entr_path/data/train.$trg.pp -max_length 60 -min_length 1 -length_ratio_threshold 1.5
mv $entr_path/data/train.en.pp.filtered $entr_path/data/train.en.pp
mv $entr_path/data/train.tr.pp.filtered $entr_path/data/train.tr.pp

echo "done!"
