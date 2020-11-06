**This code belongs to the paper "Linguistically informed masking for representation learning in the patent domain".**

It contains different folders for the different coding parts like the linguistical analysis and the domain and downstream fine-tuning of the paper.
We plan to make the patent domain fine-tuned checkpoints of the BERT and SciBERT model with the MLM and the LIM method available.
The Semantic Scholar data we used and the Wikitext-raw-2 data can be found in the datafolder. There is also a sample of the patent data we used, the whole USPTO13M dataset can be
extracted from Google BigQuery using [this query](USPTO13M_query.txt).

## Structure

- [bert](bert): [Cloned Google Research BERT Github repository](https://github.com/google-research/bert) with additional script for linguistically informed masking ([create_pretraining_data_lim.py](bert/create_pretraining_data_lim.py))
and modified script for citation prediction fine-tuning [run_classifier_cit.py](bert/run_classifier_cit.py) and bash files for [creating the domain fine-tuning data](bert/bash_create_pretraining_data), 
for [domain fine-tuning](bert/bash_further_pretrain) and for [downstream task fine-tuning](bert/bash_run_classifier)
- [citation_prediction](citation_prediction): script for [creating negative citation pairs](citation_prediction/negative_samples.py) as random permutations of the positive citation pairs, 
script for [preprocessing the citation data](citation_prediction/preprocess_cit_data.py) for fine-tuning
- [cnn_baseline](cnn_baseline): [Cloned Github repository for sentence classification using CNN](https://github.com/cahya-wirawan/cnn-text-classification-tf) following the [paper](https://www.aclweb.org/anthology/D14-1181/)
with multiple classes and pre-trained word embeddings as input, added the data preprocessing for [IPC classification](cnn_baseline/preproc_ipc.py) and [citation prediction](cnn_baseline/preproc_cit.py)
- [data](data): contains folders for [Patent](data/patent) data with data samples, and for the whole used [Semantic Scholar](data/semanticscholar) and [Wikipedia](data/wikitext) data
- [format_text_pretrain](format_text_pretrain): scripts for [creating the input format](format_text_pretrain/format_pretrain_data.py) for domain fine-tuning of BERT and for [creating the file with the noun chunk positions](format_text_pretrain/create_np_pretrain.py) 
for linguistically informed masking in domain fine-tuning and bash scripts for preprocessing multiple text files
- [IDs of patent and semantic scholar documents](IDs%20of%20patents%20and%20Semantic%20Scholar) contains the IDs of the patent and semantic scholar documents used in the different analyses, for domain fine-tuning and the downstream tasks
- [ipc_citation_dependency](ipc_citation_dependency): scripts for examining the independence of the IPC classification and citation prediction by [training a linear classifier](ipc_citation_dependency/train_classifier.py) on predicting citations given the IPC representation of the citation pair
- [ipc_classification](ipc_classification): script for [preprocessing patent data](ipc_classification/preprocess_ipc.py) for IPC classification fine-tuning
- [ling_ana](ling_ana): scripts for linguistical analysis of patent, semantic scholar and Wikipedia text: [comparison of noun chunks](ling_ana/compare_noun_chunks.py), 
[Training a patent vocabulary with sentencepiece](ling_ana/patent_vocab_train.py) and [comparing the encodings of different vocabularies](ling_ana/compare_vocab_encodings.py)
- [models](models): contains the bert and scibert model checkpoints and vocabulary as well as the [final patent vocabulary](models/sentencepiece/patents_part-000000000674_sp_preprocessed/patent_wdescr_5m_sent_30k_vocab.vocab) trained with the sentepiece algorithm
- [plots](plots): output directory of the plots created in the linguistic analysis
- [definitions.py]: NEEDS TO BE CHANGED contains the definition of the root directory of the project 
- contains the Google BigQuery queries for extracting the [UPSTO13M dataset](USPTO13M_query.txt) and the [citation dataset](citation_pairs_query.txt)

## Requirements

There are two different python environments needed for running the script, for training the BERT model we need Python2.7, for the other implementations we need Python3.
All detailed requirements can be found in [requirements_python2.txt](requirements_python2.txt) for the Python2.7 environment and in [requirements_python3.txt](requirements_python3.txt) for the Python3 environment

Python2.7 environment:
- Tensorflow==1.15

Python3 environment:
- Tensorflow > 0.12
- pandas==0.25.3
- sentencepiece==0.1.83
- spacy==2.2.2
- numpy==1.17.3

## Execution

### Preprocess Patent and Semantic Scholar text in [linguistic analysis](ling_ana)

- Run in ling_ana with Python3 environment for patent text:
    ```bash
    python preprocess_patent.py file_name
    ```
    
    where file_name needs to be the direction and name of the csv-file containing the patent text with columns 'title', 'abstract', 'claim' and 'description'.
    For example: file_name = 'data/patent/part-000000000674.csv'

- Run in ling_ana for Semantic Scholar text:
    ```bash
    python preprocess_semscho.py file_name
    ```
    
    where file_name needs to be the direction and name of the Semantic Scholar from the [Semantic Scholar Reasearch Corpus](http://s2-public-api-prod.us-west-2.elasticbeanstalk.com/corpus/?_sm_au_=iVVr4G5GHr14WkW7TRKNjKHWR8RV1)
    file containing columns 'title', 'abstract'.
    For example: file_name = 'data/semanticscholar/s2-corpus-000'


### [Linguistic analysis](ling_ana)

Run in ling_ana with Python3 environment:

- Explore IPC and CPC class occurences
    ```bash
    python ipc_cpc_class_occurences.py file_name
    ```

    where file_name needs to be the direction and name of the csv-file containing the patent text with columns 'ipc', 'cpc'.
    For example: file_name = 'data/patent/part-000000000674.csv'

- Analyze sentence length of patents
    ```bash
    python sent_length_patent.py file_name
    ```

    where file_name needs to be the direction and name of the csv-file containing the patent text with columns 'title', 'abstract', 'claim' and 'description'.
    For example: file_name = 'data/patent/part-000000000674.csv'
    
- Analyze noun chunks of patents (needs preprocessed patent text)
    ```bash
    python noun_chunks_patent.py file_name
    ```

    where file_name needs to be the direction and name of the pickle-file containing the preprocessed patent text with columns 'title', 'abstract', 'claim' and 'description'.
    For example: file_name = 'data/patent/part-000000000674_preprocessed_wo_claims.pkl'
    
- Count words in the different segments of patents (needs preprocessed patent text)
    ```bash
    python count_words_patent.py file_name
    ```

    where file_name needs to be the direction and name of the pickle-file containing the preprocessed patent text with columns 'title', 'abstract', 'claim' and 'description'.
    For example: file_name = 'data/patent/part-000000000674_preprocessed_wo_claims.pkl'

- Analyze hyphen expressions (needs preprocessed patent text)
    ```bash
    python hyphen_exp.py file_name
    ```

    where file_name needs to be the direction and name of the pickle-file containing the preprocessed patent text with columns 'title', 'abstract', 'claim' and 'description'.
    For example: file_name = 'data/patent/part-000000000674_preprocessed_wo_claims.pkl'

- Analyze the Semantic Scholar text (needs preprocessed semantic scholar text)
    ```bash
    python ling_ana_semscho.py file_name
    ```

    where file_name needs to be the direction and name of the pickle-file containing the preprocessed Semantic Scholar text with columns 'title', 'abstract'.
    For example: file_name = 'data/semanticscholar/s2-corpus-000_clean.pkl'
    
- Analyze the Wikipedia dataset
    ```bash
    python ling_ana_wiki.py location_input location_output
    ```

    where location_input needs to be the direction and name of the input Wikitext raw 2 file and location_output is the direction where the preprocessed file will be saved.
    For example: location_input = 'data/wikitext/wikitext-2-raw/wiki.test.raw', location_output = 'data/wikitext/wikitext-2-raw/wiki.train.wo.captions.txt'

- Compare the noun chunk distributions of patent, semantic scholar and wikipedia data (needs preprocessed patent and semantic scholar text)
    ```bash
    python compare_noun_chunks.py patent_file semscho_file wiki_file
    ```

    where patent_file and semscho need to be the direction and name of the preprocessed patent or semantic scholar pickle file, and the direction of the Wikipedia file
    For example: patent_file = 'data/en_since_2000_a_unzip/part-000000000674_preprocessed_wo_claims.pkl', sem_file = 'data/semanticscholar/s2-corpus-000_clean.pkl', wiki_file = 'data/wikitext/wikitext-2-raw/wiki.train.raw'

- Training a patent vocabulary
    ```bash
    python patent_vocab_train.py patent_file model_dir
    ```

    where patent_file needs to be the direction and name of the csv-file containing the patent text and model_dir is the direction of the model output
    For example: patent_file = 'data/patent/part-000000000674.csv', model_dir = 'models/sentencepiece/patents_part-000000000674_sp_preprocessed/patent_wdescr_50k_sent_30k_vocab'

- Comparing the different vocabularies for encoding patent text
    ```bash
    python compare_vocab_encodings.py patent_file model_dir bert_tokenizer scibert_tokenizer
    ```

    where patent_file needs to be the direction and name of the preprocessed pickle file containing the patent text and model_dir is the direction of the sentencepiece model,
    bert_tokenizer is the direction of the bert_vocabulary and scibert_tokenizer is the direction of the scibert vocabulary
    For example: file_name = 'data/patent/part-000000000674_preprocessed.pkl', model_dir = 'models/sentencepiece/patents_part-000000000674_sp_preprocessed/patent_wdescr_5m_sent_30k_vocab'
    , bert_tokenizer = ,'bert-base-cased', scibert_tokenizer = 'models/scibert_scivocab_cased'
    
    
### [Format patent text for domain fine-tuning](format_text_pretrain)

Run in format_text_pretrain in Python2.7 environment:

- Create the format for the domain fine-tuning with BERT
     ```bash
    python format_pretrain_data.py file_loc file_name
    ```

    where file_loc is the directory of the files and file_name is the name of the csv.gz files containing the patent text
    For example: file_location = '/home/ubuntu/Documents/thesis/data/patent/', file_name = 'part-000000000674.csv.gz'

- Create the file with the noun chunk positions
    ```bash
     python create_np_pretrain.py input_file output_file 
    ```

    where input_file_loc is the text file from the output of format_pretrain_data and output_file is the name of the output_file with the noun chunk positions denoted with a value > 1.
    For example: input_file = 'bert/data/npvector_test_text.txt',  output_file = 'bert/data/npvector_test_vectors.txt'


### [Linguistically informed masking](bert)

Run in bert in Python2.7 environment:

- Create pretraining data for training BERT with the LIM method:
    ```bash
  export BERT_BASE_DIR=/path/to/bert/cased_L-12_H-768_A-12
  python create_pretraining_data_lim.py   
  --input_file=./data/part-000000000674.txt 
  --input_file_np=./data/np_vectors_part-000000000674.txt  
  --output_file=./data/tfrecord_128_lim_1_part-000000000674.tfrecord 
  --lim_prob=1.0 
  --vocab_file=$BERT_BASE_DIR/vocab.txt   
  --do_lower_case=False   
  --max_seq_length=128   
  --max_predictions_per_seq=20   
  --masked_lm_prob=0.15   
  --random_seed=12345 
  --dupe_factor=5
    ```

    where lim_prob is the noun chunk masking probability and input_file_np is the input file of the noun chunking masking position txt file
    
For domain-finetuning with MLM and for fine-tuning on the downstream tasks, we use the same commands as in the Google repository and they can be found in the bash scripts.

### [IPC classification](ipc_classification)

Run in ipc_classifcation in Python3 environment:

- Preprocess data for IPC classification
    ```bash
     python preprocess_ipc.py file_location new_file_location train_test start_index end_index 
    ```

    where file_location is the directory of the csv.gv-files with the patent text, new_file_location is the directory where the new files are stored,
    train_test determines if the files belong to the train or test set and needs to be either 'train' or 'test', start_index is the start number of the files and end_index is the end number of the files
    where the number is in the file name for example 'part-00000000674.csv.gz'
    For example: file_location = '/home/ubuntu/Documents/thesis/data/patent_contents_en_since_2000_application_kind_a'
    new_file_location = '/home/ubuntu/PycharmProjects/patent/data/ipc'
    train_test = 'train'
    start_files = 670
    end_files = 674
    
### [Citation prediction](citation_prediction)

Run in citation_precition in Python3 environment:

- Preprocess data
    ```bash
     python preprocess_data.py
    ```
    preprocesses the claim data of the positive pairs and creates a file 'patent-contents-for-citations_en_claim_wclaims_all.pkl'

- Create negative citation pairs with ramdon permutation of positive citation pairs
    ```bash
     python negative_samples.py df_location positive_pairs_loc
    ```
    where df_location is the location of the preprocessed positive citation pairs created above and positive_pairs_loc is
    the location of the csv file containing the claims
    For example: df_location = "data/citations/patent-contents-for-citations-wclaims/patent-contents-for-citations_en_claim_wclaims_all.pkl"
    positive_pairs_loc = 'data/citations/patent-contents-for-citations-wclaims/citations-only-type-x-with_claims.csv'
    
- Preprocess citation pairs for fine-tuning BERT
    ```bash
     python preprocess_cit_data.py file_name start_index end_index
    ```
    where file_name is the name of the csv.gz file containing the patent text, start_index is the number of the patent document and end_index is the end number of the documents
    For example: file_name = 'data/citations/patent-contents-for-citations-wclaims/patent-contents-for-citations_en_claim_wclaims000000000000.csv.gz'
    start_index = 0
    end_index = 10
    
### [cnn_baseline](cnn_baseline)

Run in cnn_baseline with Python3 environment:

- Preprocess the IPC classification data
    ```bash
     python preprocess_ipc.py input_dir output_dir start_index end_index
    ```
    where input_dir is the input directory containing the tsv files which are used for the IPC classification downstream in BERT fine-tuning
    output_dir is the output directory where a structure of folders is built with each sample being a text file contained in the folder of its IPC tag
    start_index is the start number of the files, end_index is the end number of the files
    For example: input_dir = /data/ipc_classification/'
    output_dir = '/data/cnn_baseline/ipc/'
    start_index = 0
    end_index = 10

- Preprocess the citation data
    ```bash
     python preprocess_cit.py file_name sample_number output_dir
    ```
    where file_name is the name of the pickle file containing all citation pairs
    sample__number is the number of samples which are taken from the file with the citation pairs
    output_dir is the output directory where a file with the positive pairs and a file with the negative pairs is created
    For example: file_name = 'data/citations/patent-contents-for-citations-wclaims/citations-only-type-x-with_claims_train_data.pkl'
    number_of_samples = 16000
    output_dir = 'data/cnn_baseline/citation/'

- Train the CNN
    ```bash
    ./train.py
    ```

- Evaluate the CNN
    ```bash
    ./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
    ```
    Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.

    
### [IPC and citation dependency](ipc_citation_dependency)

Run in ipc_citation_dependency with Python3 environment:

- Preprocess data for training the linear classifier in preprocess_train_test.py
- Train a linear classifier using SciKitlearn in train_classifier.py
