# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
import os
import math
import nltk
import spacy
import sys


def preprocess_text_column(df: pd.DataFrame):
    df['text'] = df[['title', 'abstract', 'claim', 'description']].apply(lambda x: '\n'.join(x), axis=1)
    # Remove multiple whitespaces from text and tabs and newlines :(
    df['text'] = df['text'].str.replace('[ \t]+', ' ', regex=True)
    # Replace multiple \n by \n
    df['text'] = df['text'].str.replace('\n\s{1,10}', '\n ', regex=True)
    # Add a newline at the end of each patent document
    #df['text'] = df['text'].astype(str) + '[DOCEND]'
    return df


def split_sentences(column: pd.Series, file_location, file_name):
    nlp = spacy.blank("en")
    nlp.add_pipe(nlp.create_pipe("sentencizer"))

    full_text_sents = []
    for doc in nlp.pipe(column):
        full_text_sents.extend(sent.text.strip() for sent in doc.sents)
    with open(os.path.join(file_location, '{0}.txt'.format(file_name.split('.')[0])), 'w') as f:
        f.write('\n'.join(sent for sent in full_text_sents).replace('\n\n', '\n').replace('[DOCEND]', '\n'))


def split_sentences_nltk(column:pd.Series, file_location, file_name):
    patent_number_per_txt_file = math.ceil(len(column)/10)
    # Splitte text file auf in 10 untertext files
    for i in range(9):
        with open(os.path.join(file_location, '{0}{1}.txt'.format(file_name.split('.')[0], i)), 'w') as f:
            for text in column[i*patent_number_per_txt_file:(i+1)*patent_number_per_txt_file]:
                sentences = nltk.tokenize.sent_tokenize(text)
                f.write('\n'.join(sent for sent_unsplitted in sentences for sent in sent_unsplitted.split('\n') if len(sent.split()) > 2))  # .replace('\n\n', '\n') vor replace docend #.replace('[DOCEND]', '\n\n')
                f.write('\n\n')
    # damit index von letztem text file nicht out of range ist
    i = 9
    with open(os.path.join(file_location, '{0}{1}.txt'.format(file_name.split('.')[0], i)), 'w') as f:
        for text in column[i * patent_number_per_txt_file:]:
            sentences = nltk.tokenize.sent_tokenize(text)
            f.write('\n'.join(sent for sent_unsplitted in sentences for sent in sent_unsplitted.split('\n') if len(
                sent.split()) > 2))  # .replace('\n\n', '\n') vor replace docend #.replace('[DOCEND]', '\n\n')
            f.write('\n\n')


def main(file_location, file_name):
    df = pd.read_csv(os.path.join(file_location, file_name), compression='gzip', error_bad_lines=False)
    df = preprocess_text_column(df)
    split_sentences_nltk(df['text'], file_location, file_name)


if __name__ == '__main__':
    file_location = sys.argv[1]
    file_name = sys.argv[2]
    #file_location = '/home/ubuntu/Documents/thesis/data/patent_contents_en_since_2000_application_kind_a'
    #file_name = 'part-000000000674.csv.gz'
    main(str(file_location), str(file_name))












