#!/bin/bash
#
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
# following mapping of encoding: {'acl': 2, 'acomp': 3, 'advcl': 4, 'advmod': 5, 'agent': 6, 'amod': 7, 'appos': 8, 'attr': 9, 'aux': 10, 'auxpass': 11, 'case': 12, 'cc': 13, 'ccomp': 14, 'clf': 15, 'compound': 16, 'conj': 17, 'cop': 18, 'csubj': 19, 'csubjpass': 20, 'dative': 21, 'dep': 22, 'det': 23, 'discourse': 24, 'dislocated': 25, 'dobj': 26, 'expl': 27, 'fixed': 28, 'flat': 29, 'goeswith': 30, 'iobj': 31, 'intj': 32, 'list': 33, 'mark': 34, 'meta': 35, 'neg': 36, 'nn': 37, 'nmod': 38, 'nounmod': 39, 'npadvmod': 40, 'npmod': 41, 'nsubj': 42, 'nsubjpass': 43, 'nummod': 44, 'oprd': 45, 'obj': 46, 'obl': 47, 'orphan': 48, 'parataxis': 49, 'pcomp': 50, 'pobj': 51, 'poss': 52, 'preconj': 53, 'predet': 54, 'prep': 55, 'prt': 56, 'punct': 57, 'quantmod': 58, 'relcl': 59, 'reparandum': 60, 'root': 61, 'vocative': 62, 'xcomp': 63, '': 64}
trap "exit" INT
source activate tensorflow_p27
#runList='6700 6701 6702 6703 6704 6705 6706 6707 6708 6709 6720 6721 6722 6723 6724 6725 6726 6727 6728 6729'
foo () {
    local run=$1
   # screen
   # source activate tensorflow_p27
    aws s3 cp s3://aws-s3-patent/tfrecords_mlm/tfrecord_128_part-000000000$run.tfrecord /home/ubuntu/bert/data/tfrecord_128_part-000000000$run.tfrecord
   # export KMP_AFFINITY=none
   # export BERT_BASE_DIR=/home/ubuntu/bert
   #  aws s3 cp /home/ubuntu/bert/data/tfrecords_seq_length128/tfrecord_128_part-000000000$run.tfrecord s3://aws-s3-patent/tfrecords_mlm/tfrecord_128_part-000000000$run.tfrecord
   # rm data/part-000000000$run.txt
  echo "Downloaded tfrecords of sample $run" -e
}
for run in {6550..6709}; do foo "$run" & done
wait
for run in {6720..6749}; do foo "$run" & done
wait
for run in {6550..6709};
do
 DESTINATIONS+="./data/tfrecord_128_part-000000000${run}.tfrecord,"
done
for run in {6720..6749};
do
 DESTINATIONS+="./data/tfrecord_128_part-000000000${run}.tfrecord,"
done
DESTINATIONS=${DESTINATIONS::-1}
echo $DESTINATIONS
wait
export BERT_BASE_DIR=/home/ubuntu/bert
python run_pretraining.py --input_file=$DESTINATIONS  --output_dir=$BERT_BASE_DIR/output/bert_mlm   --do_train=True   --do_eval=True   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --save_checkpoints_steps=5000   --train_batch_size=32   --max_seq_length=128   --max_predictions_per_seq=20   --num_train_steps=100000   --num_warmup_steps=10000 --learning_rate=5e-5
wait
aws s3 cp --recursive /home/ubuntu/bert/output/bert_mlm/ s3://aws-s3-patent/models/bert/further_5e05_100k/
wait
aws ec2 stop-instances --instance-ids=i-08741030e81b4c63a --region=eu-central-1

#export -f foo
#parallel foo ::: {6650..6652} 
