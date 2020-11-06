#!/bin/bash
source ./bert_env8/bin/activate
export BERT_BASE_DIR=/home/ubuntu/PycharmProjects/patent/bert
runList='667 668 669 670 672 673 674'
foo () {
    local run=$1
      python create_pretraining_data.py \
  --input_file=./data/part-000000000$run.txt \
  --output_file=./data/tf_examples$run.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=False \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
  echo "Wrote tf records of sample $i"
}
for run in $runList; do foo "$run" & done
