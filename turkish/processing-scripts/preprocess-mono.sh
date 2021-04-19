#!/bin/bash
set -eo pipefail

moses_scripts="/home/laurie/apps/mosesdecoder/scripts"
mono_path="/mnt/startiger0/laurie/diversity/turkish-baseline/mono_data"
tc_folder="/mnt/startiger0/laurie/diversity/turkish-baseline/parallel_data/model"
tr_mono="$mono_path/news.tr"
en_mono="$mono_path/news.en.2014-2017"


echo "Normalising punctuation"
$moses_scripts/tokenizer/normalize-punctuation.perl -l en < $en_mono > $en_mono.norm
$moses_scripts/tokenizer/normalize-punctuation.perl -l tr < $tr_mono > $tr_mono.norm

echo "Truecasing"
$moses_scripts/recaser/truecase.perl < $en_mono.norm > $en_mono.pp -model $tc_folder/truecase-model.en
$moses_scripts/recaser/truecase.perl < $tr_mono.norm > $tr_mono.pp -model $tc_folder/truecase-model.tr

rm $en_mono.norm $de_mono.norm $tr_mono.norm

echo "done"
