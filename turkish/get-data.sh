#/bin/bash
MOSES_DECODER="/home/laurie/apps/mosesdecoder"


# download data
wget http://data.statmt.org/wmt18/translation-task/dev.tgz
tar zxvf dev.tgz

wget http://data.statmt.org/wmt18/translation-task/test.tgz
tar zxvf test.tgz

wget http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz
tar zxvf training-parallel-nc-v13.tgz
mv training-parallel-nc-v13 train

rm *.tgz

# get rid of all this sgm stuff
mkdir wmt_entr

wget http://opus.nlpl.eu/download.php?f=SETIMES/v2/moses/en-tr.txt.zip -O en-tr.zip
unzip en-tr.zip -d wmt_entr
rm en-tr.zip wmt_entr/LICENSE wmt_entr/README wmt_entr/SETIMES.en-tr.ids


# train
mv wmt_entr/SETIMES.en-tr.en wmt_entr/train.en
mv wmt_entr/SETIMES.en-tr.tr wmt_entr/train.tr

# dev
$MOSES_DECODER/scripts/ems/support/input-from-sgm.perl < dev/newstest2017-tren-src.tr.sgm > wmt_entr/dev.tr
$MOSES_DECODER/scripts/ems/support/input-from-sgm.perl < dev/newstest2017-tren-ref.en.sgm > wmt_entr/dev.en

# test
$MOSES_DECODER/scripts/ems/support/input-from-sgm.perl < test/newstest2018-tren-src.tr.sgm > wmt_entr/test.tr
$MOSES_DECODER/scripts/ems/support/input-from-sgm.perl < test/newstest2018-tren-ref.en.sgm > wmt_entr/test.en

rm -rf dev/ train/ test/
