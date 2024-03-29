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
############## namen einsammeln
for run in {181..190};
do
 DESTINATIONS2+="./data/ipc_claim_scibert_dev_0${run}.tfrecord,"
done
DESTINATIONS2=${DESTINATIONS2::-1}
echo $DESTINATIONS2
wait
export BERT_BASE_DIR=/home/ubuntu/bert
export DATA_DIR=/home/ubuntu/bert/data
################## Bert Vanilla 
python run_classifier.py  --task_name=IPC  --do_train=false  --do_eval=true --input_train_file=$DESTINATIONS --input_eval_file=$DESTINATIONS2  --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/scibert_model.ckpt  --save_checkpoints_steps=5000 --max_seq_length=128  --train_batch_size=32   --learning_rate=5e-5  --num_train_epochs=1.0 --num_train_steps=60000  --do_lower_case=False   --label_location=$DATA_DIR/ipc_all_unique_tags_files_000-125_180-200.pkl  --output_dir=$BERT_BASE_DIR/output/ipc_classification/scibert_vanilla/
wait
aws s3 cp --recursive /home/ubuntu/bert/output/ipc_classification/scibert_vanilla/ s3://aws-s3-patent/downstream_tasks/ipc_classification/models/scibert/scibert_vanilla/
wait
################## Bert MLM
python run_classifier.py  --task_name=IPC  --do_train=false  --do_eval=true --input_train_file=$DESTINATIONS --input_eval_file=$DESTINATIONS2  --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/model_scibert_mlm.ckpt-100000 --save_checkpoints_steps=5000  --max_seq_length=128  --train_batch_size=32   --learning_rate=5e-5  --num_train_epochs=1.0 --num_train_steps=60000  --do_lower_case=False   --label_location=$DATA_DIR/ipc_all_unique_tags_files_000-125_180-200.pkl  --output_dir=$BERT_BASE_DIR/output/ipc_classification/scibert_mlm/
wait
aws s3 cp --recursive /home/ubuntu/bert/output/ipc_classification/scibert_mlm/ s3://aws-s3-patent/downstream_tasks/ipc_classification/models/scibert/scibert_mlm/
wait
################## Bert LIM1
python run_classifier.py  --task_name=IPC  --do_train=false  --do_eval=true --input_train_file=$DESTINATIONS --input_eval_file=$DESTINATIONS2  --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/model_scibert_lim1.ckpt-100000 --save_checkpoints_steps=5000  --max_seq_length=128  --train_batch_size=32   --learning_rate=5e-5  --num_train_epochs=1.0 --num_train_steps=60000  --do_lower_case=False   --label_location=$DATA_DIR/ipc_all_unique_tags_files_000-125_180-200.pkl  --output_dir=$BERT_BASE_DIR/output/ipc_classification/scibert_lim1/
wait
aws s3 cp --recursive /home/ubuntu/bert/output/ipc_classification/scibert_lim1/ s3://aws-s3-patent/downstream_tasks/ipc_classification/models/scibert/scibert_lim1/
wait
################## Bert LIM075
python run_classifier.py  --task_name=IPC  --do_train=false  --do_eval=true --input_train_file=$DESTINATIONS --input_eval_file=$DESTINATIONS2  --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/model_scibert_lim075.ckpt-100000 --save_checkpoints_steps=5000 --max_seq_length=128  --train_batch_size=32   --learning_rate=5e-5  --num_train_epochs=1.0 --num_train_steps=60000  --do_lower_case=False   --label_location=$DATA_DIR/ipc_all_unique_tags_files_000-125_180-200.pkl  --output_dir=$BERT_BASE_DIR/output/ipc_classification/scibert_lim075/
wait
aws s3 cp --recursive /home/ubuntu/bert/output/ipc_classification/scibert_lim075/ s3://aws-s3-patent/downstream_tasks/ipc_classification/models/scibert/scibert_lim075/
wait
aws ec2 stop-instances --instance-ids=i-03043a582d03b31a6 --region=eu-central-1
