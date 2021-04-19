#!/bin/bash
#
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
source ./bert_env8/bin/activate
export BERT_BASE_DIR=/home/ubuntu/PycharmProjects/patent/bert
echo "Activated environment"
python create_pretraining_data.py \
  --input_file=./data/first1000part674.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=False \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
echo "Created pretraining data"
