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
#for run in {000..004};
#do
# DESTINATIONS+="./data/ipc_claim_scibert_train_0${run}.tfrecord,"
#done
#DESTINATIONS=${DESTINATIONS::-1}
#echo $DESTINATIONS
wait
export BERT_BASE_DIR=/home/ubuntu/bert
export DATA_DIR=/home/ubuntu/bert/data
################## Bert Vanilla 
python run_classifier_cit.py  --task_name=CIT  --do_train=false  --do_eval=true --input_train_file=./data/citations-only-type-x-with_claims_train_data_scibert.tfrecord  --input_eval_file=./data/citations-only-type-x-with_claims_eval_data_2000.tfrecord   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/output/cit_prediction/scibert_vanilla2/model.ckpt-250  --max_seq_length=256  --eval_batch_size=16   --learning_rate=2e-5  --num_train_epochs=1.0 --num_train_steps=750 --save_checkpoints_steps=250  --do_lower_case=False    --output_dir=$BERT_BASE_DIR/output/cit_prediction_eval/scibert_vanilla/steps250/
wait
aws s3 cp  /home/ubuntu/bert/output/cit_prediction_eval/scibert_vanilla/steps250/predictions_eval.pkl s3://aws-s3-patent/downstream_tasks/citations/evaluation/scibert/scibert_vanilla/predictions_eval250.pkl
wait
#python run_classifier_cit.py  --task_name=CIT  --do_train=false  --do_eval=true --input_train_file=./data/citations-only-type-x-with_claims_train_data_scibert.tfrecord  --input_eval_file=./data/citations-only-type-x-with_claims_eval_data_2000.tfrecord   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/output/cit_prediction/scibert_vanilla2/model.ckpt-500  --max_seq_length=256  --eval_batch_size=16   --learning_rate=2e-5  --num_train_epochs=1.0 --num_train_steps=750 --save_checkpoints_steps=250  --do_lower_case=False    --output_dir=$BERT_BASE_DIR/output/cit_prediction_eval/scibert_vanilla/steps500/
#wait
#aws s3 cp  /home/ubuntu/bert/output/cit_prediction_eval/scibert_vanilla/steps500/eval_results.txt s3://aws-s3-patent/downstream_tasks/citations/evaluation/scibert/scibert_vanilla/eval_resultssteps500.txt
#wait
python run_classifier_cit.py  --task_name=CIT  --do_train=false  --do_eval=true --input_train_file=./data/citations-only-type-x-with_claims_train_data_scibert.tfrecord  --input_eval_file=./data/citations-only-type-x-with_claims_eval_data_2000.tfrecord   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/output/cit_prediction/scibert_vanilla2/model.ckpt-750  --max_seq_length=256  --eval_batch_size=16   --learning_rate=2e-5  --num_train_epochs=1.0 --num_train_steps=750 --save_checkpoints_steps=250  --do_lower_case=False    --output_dir=$BERT_BASE_DIR/output/cit_prediction_eval/scibert_vanilla/steps750/
wait
aws s3 cp  /home/ubuntu/bert/output/cit_prediction_eval/scibert_vanilla/steps750/predictions_eval.pkl s3://aws-s3-patent/downstream_tasks/citations/evaluation/scibert/scibert_vanilla/predictions_eval750.pkl
wait
################## Bert MLM
python run_classifier_cit.py  --task_name=CIT  --do_train=false  --do_eval=true --input_train_file=./data/citations-only-type-x-with_claims_train_data_scibert.tfrecord  --input_eval_file=./data/citations-only-type-x-with_claims_eval_data_2000.tfrecord   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/output/cit_prediction/scibert_mlm/model.ckpt-250   --max_seq_length=256  --eval_batch_size=16   --learning_rate=2e-5  --num_train_epochs=1.0 --num_train_steps=1000 --save_checkpoints_steps=250 --do_lower_case=False     --output_dir=$BERT_BASE_DIR/output/cit_prediction_eval/scibert_mlm/steps250/
wait
aws s3 cp /home/ubuntu/bert/output/cit_prediction_eval/scibert_mlm/steps250/predictions_eval.pkl s3://aws-s3-patent/downstream_tasks/citations/evaluation/scibert/scibert_mlm/predictions_eval250.pkl
wait
#python run_classifier_cit.py  --task_name=CIT  --do_train=false  --do_eval=true --input_train_file=./data/citations-only-type-x-with_claims_train_data_scibert.tfrecord  --input_eval_file=./data/citations-only-type-x-with_claims_eval_data_2000.tfrecord   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/output/cit_prediction/scibert_mlm/model.ckpt-500   --max_seq_length=256  --eval_batch_size=16   --learning_rate=2e-5  --num_train_epochs=1.0 --num_train_steps=1000 --save_checkpoints_steps=250 --do_lower_case=False     --output_dir=$BERT_BASE_DIR/output/cit_prediction_eval/scibert_mlm/steps500/
#wait
#aws s3 cp /home/ubuntu/bert/output/cit_prediction_eval/scibert_mlm/steps500/eval_results.txt s3://aws-s3-patent/downstream_tasks/citations/evaluation/scibert/scibert_mlm/eval_resultssteps500.txt
#wait
python run_classifier_cit.py  --task_name=CIT  --do_train=false  --do_eval=true --input_train_file=./data/citations-only-type-x-with_claims_train_data_scibert.tfrecord  --input_eval_file=./data/citations-only-type-x-with_claims_eval_data_2000.tfrecord   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/output/cit_prediction/scibert_mlm/model.ckpt-750   --max_seq_length=256  --eval_batch_size=16   --learning_rate=2e-5  --num_train_epochs=1.0 --num_train_steps=1000 --save_checkpoints_steps=250 --do_lower_case=False     --output_dir=$BERT_BASE_DIR/output/cit_prediction_eval/scibert_mlm/steps750/
wait
aws s3 cp /home/ubuntu/bert/output/cit_prediction_eval/scibert_mlm/steps750/predictions_eval.pkl s3://aws-s3-patent/downstream_tasks/citations/evaluation/scibert/scibert_mlm/predictions_eval750.pkl
wait
################## Bert LIM1
python run_classifier_cit.py  --task_name=CIT  --do_train=false  --do_eval=true --input_train_file=./data/citations-only-type-x-with_claims_train_data_scibert.tfrecord  --input_eval_file=./data/citations-only-type-x-with_claims_eval_data_2000.tfrecord   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/output/cit_prediction/scibert_lim1/model.ckpt-250   --max_seq_length=256  --eval_batch_size=16   --learning_rate=2e-5  --num_train_epochs=1.0 --num_train_steps=1000 --save_checkpoints_steps=250 --do_lower_case=False    --output_dir=$BERT_BASE_DIR/output/cit_prediction_eval/scibert_lim1/steps250/
wait
aws s3 cp /home/ubuntu/bert/output/cit_prediction_eval/scibert_lim1/steps250/predictions_eval.pkl s3://aws-s3-patent/downstream_tasks/citations/evaluation/scibert/scibert_lim1/predictions_eval250.pkl
wait
#python run_classifier_cit.py  --task_name=CIT  --do_train=false  --do_eval=true --input_train_file=./data/citations-only-type-x-with_claims_train_data_scibert.tfrecord  --input_eval_file=./data/citations-only-type-x-with_claims_eval_data_2000.tfrecord   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/output/cit_prediction/scibert_lim1/model.ckpt-500   --max_seq_length=256  --eval_batch_size=16   --learning_rate=2e-5  --num_train_epochs=1.0 --num_train_steps=1000 --save_checkpoints_steps=250 --do_lower_case=False    --output_dir=$BERT_BASE_DIR/output/cit_prediction_eval/scibert_lim1/steps500/
#wait
#aws s3 cp /home/ubuntu/bert/output/cit_prediction_eval/scibert_lim1/steps500/eval_results.txt s3://aws-s3-patent/downstream_tasks/citations/evaluation/scibert/scibert_lim1/eval_resultssteps500.txt
#wait
python run_classifier_cit.py  --task_name=CIT  --do_train=false  --do_eval=true --input_train_file=./data/citations-only-type-x-with_claims_train_data_scibert.tfrecord  --input_eval_file=./data/citations-only-type-x-with_claims_eval_data_2000.tfrecord   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/output/cit_prediction/scibert_lim1/model.ckpt-750   --max_seq_length=256  --eval_batch_size=16   --learning_rate=2e-5  --num_train_epochs=1.0 --num_train_steps=1000 --save_checkpoints_steps=250 --do_lower_case=False    --output_dir=$BERT_BASE_DIR/output/cit_prediction_eval/scibert_lim1/steps750/
wait
aws s3 cp /home/ubuntu/bert/output/cit_prediction_eval/scibert_lim1/steps750/predictions_eval.pkl s3://aws-s3-patent/downstream_tasks/citations/evaluation/scibert/scibert_lim1/predictions_eval750.pkl
wait
################## Bert LIM075
python run_classifier_cit.py  --task_name=CIT  --do_train=false  --do_eval=true --input_train_file=./data/citations-only-type-x-with_claims_train_data_scibert.tfrecord --input_eval_file=./data/citations-only-type-x-with_claims_eval_data_2000.tfrecord   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/output/cit_prediction/scibert_lim0752/model.ckpt-250   --max_seq_length=256  --eval_batch_size=16   --learning_rate=2e-5  --num_train_epochs=1.0 --num_train_steps=750 --save_checkpoints_steps=250  --do_lower_case=False   --output_dir=$BERT_BASE_DIR/output/cit_prediction_eval/scibert_lim075/steps250/
wait
aws s3 cp /home/ubuntu/bert/output/cit_prediction_eval/scibert_lim075/steps250/predictions_eval.pkl s3://aws-s3-patent/downstream_tasks/citations/evaluation/scibert/scibert_lim075/predictions_eval250.pkl
wait
#python run_classifier_cit.py  --task_name=CIT  --do_train=false  --do_eval=true --input_train_file=./data/citations-only-type-x-with_claims_train_data_scibert.tfrecord --input_eval_file=./data/citations-only-type-x-with_claims_eval_data_2000.tfrecord   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/output/cit_prediction/scibert_lim0752/model.ckpt-500   --max_seq_length=256  --eval_batch_size=16   --learning_rate=2e-5  --num_train_epochs=1.0 --num_train_steps=750 --save_checkpoints_steps=250  --do_lower_case=False   --output_dir=$BERT_BASE_DIR/output/cit_prediction_eval/scibert_lim075/steps500/
#wait
#aws s3 cp /home/ubuntu/bert/output/cit_prediction_eval/scibert_lim075/steps500/eval_results.txt s3://aws-s3-patent/downstream_tasks/citations/evaluation/scibert/scibert_lim075/eval_resultssteps500.txt
#wait
python run_classifier_cit.py  --task_name=CIT  --do_train=false  --do_eval=true --input_train_file=./data/citations-only-type-x-with_claims_train_data_scibert.tfrecord --input_eval_file=./data/citations-only-type-x-with_claims_eval_data_2000.tfrecord   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/scibert_vocab.txt  --bert_config_file=$BERT_BASE_DIR/scibert_config.json --init_checkpoint=$BERT_BASE_DIR/output/cit_prediction/scibert_lim0752/model.ckpt-750   --max_seq_length=256  --eval_batch_size=16   --learning_rate=2e-5  --num_train_epochs=1.0 --num_train_steps=750 --save_checkpoints_steps=250  --do_lower_case=False   --output_dir=$BERT_BASE_DIR/output/cit_prediction_eval/scibert_lim075/steps750/
wait
aws s3 cp /home/ubuntu/bert/output/cit_prediction_eval/scibert_lim075/steps750/predictions_eval.pkl s3://aws-s3-patent/downstream_tasks/citations/evaluation/scibert/scibert_lim075/predictions_eval750.pkl
wait
aws ec2 stop-instances --instance-ids=i-03043a582d03b31a6 --region=eu-central-1
