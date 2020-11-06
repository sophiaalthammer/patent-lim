#!/bin/bash
# following mapping of encoding: {'acl': 2, 'acomp': 3, 'advcl': 4, 'advmod': 5, 'agent': 6, 'amod': 7, 'appos': 8, 'attr': 9, 'aux': 10, 'auxpass': 11, 'case': 12, 'cc': 13, 'ccomp': 14, 'clf': 15, 'compound': 16, 'conj': 17, 'cop': 18, 'csubj': 19, 'csubjpass': 20, 'dative': 21, 'dep': 22, 'det': 23, 'discourse': 24, 'dislocated': 25, 'dobj': 26, 'expl': 27, 'fixed': 28, 'flat': 29, 'goeswith': 30, 'iobj': 31, 'intj': 32, 'list': 33, 'mark': 34, 'meta': 35, 'neg': 36, 'nn': 37, 'nmod': 38, 'nounmod': 39, 'npadvmod': 40, 'npmod': 41, 'nsubj': 42, 'nsubjpass': 43, 'nummod': 44, 'oprd': 45, 'obj': 46, 'obl': 47, 'orphan': 48, 'parataxis': 49, 'pcomp': 50, 'pobj': 51, 'poss': 52, 'preconj': 53, 'predet': 54, 'prep': 55, 'prt': 56, 'punct': 57, 'quantmod': 58, 'relcl': 59, 'reparandum': 60, 'root': 61, 'vocative': 62, 'xcomp': 63, '': 64}
trap "exit" INT
#source activate tensorflow_p27
export BERT_BASE_DIR=/home/ubuntu/bert
foo () {
    local run=$1
    python create_pretraining_data.py   --input_file=./data/part-000000000$run.txt   --output_file=./data/tfrecords_seq_length128/tfrecords_128_part-000000000$run.tfrecord   --vocab_file=$BERT_BASE_DIR/vocab.txt   --do_lower_case=False   --max_seq_length=128   --max_predictions_per_seq=20   --masked_lm_prob=0.15   --random_seed=12345 --dupe_factor=5
     aws s3 cp /home/ubuntu/bert/data/tfrecords_seq_length128/tfrecords_128_part-000000000$run.tfrecord s3://aws-s3-patent/tfrecords_mlm/tfrecord_128_part-000000000$run.tfrecord
  echo "Wrote tfrecords of sample $run" -e
}
for run in {6730..6749}; do foo "$run" & done
