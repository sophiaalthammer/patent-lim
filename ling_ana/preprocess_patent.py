# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
import os
import sys
from ling_ana import sent_length_patent
from definitions import ROOT_DIR


def preprocess_column(column: pd.Series):
    """
    Preprocesses text in a whole column by removing short sentence blocks (less than 10 words for all sentences in block)
    :param column: pd.Series containing text per cell
    :return: None
    """
    sep = ' '
    for i in range(len(column)):
        sentences = sent_length_patent.get_sentences(column.iloc[i])
        lengths = sent_length_patent.count_sentence_length(sentences)
        sent_short, leng_short = sent_length_patent.remove_short_sentence_blocks(sentences, lengths)
        column[i] = sep.join(sent_short)


def preprocess_column_short_sentences(column: pd.Series, limit: int):
    """
        Preprocesses text in a whole column by removing short sentence blocks (less than 10 words for all sentences in block)
        :param column: pd.Series containing text per cell
        :return: None
        """
    sep = ' '
    for i in range(len(column)):
        sentences = sent_length_patent.get_sentences(column.iloc[i])
        lengths = sent_length_patent.count_sentence_length(sentences)
        sent_short, leng_short = sent_length_patent.remove_short_sentences(sentences, lengths, limit)
        column[i] = sep.join(sent_short)


def preprocess_column_long_sentences(column: pd.Series, limit: int):
    """
    Preprocesses text in a whole column by removing long sentences (more than 250 words)
    :param column: pd.Series containing text per cell
    :return: None
    """
    sep = ' '
    for i in range(len(column)):
        sentences = sent_length_patent.get_sentences(column.iloc[i])
        lengths = sent_length_patent.count_sentence_length(sentences)
        sent_short, leng_short = sent_length_patent.remove_long_sentences(sentences, lengths, limit)
        column[i] = sep.join(sent_short)


def main(ROOT_DIR, file_name):
    """
    Preprocess patent text by removing short sentence blocks, requires a csv input format with text columns abstract,
    claim and description, saves the dataframe with the preprocessed text as pickle file
    :param ROOT_DIR: root directory of data folder
    :param file_name: file name of the csv file
    :return: None
    """
    # Preprocess texts for linguistic analysis: remove short sentence blocks
    df = pd.read_csv(os.path.join(ROOT_DIR, file_name))
    preprocess_column(df['abstract'])
    preprocess_column(df['claim'])
    preprocess_column(df['description'])
    # Add a new column text with the full text of the patent
    df['text'] = df[['title', 'abstract', 'claim', 'description']].apply(lambda x: '\n'.join(x), axis=1)
    # Save file
    df.to_pickle(os.path.join(ROOT_DIR, '{0}_preprocessed_wo_claims.pkl'.format(file_name.split('.')[0])))

    # remove sentences shorter than 3 for claims, because the correspond to enumerations of the claims
    #  (1 . as sentence of length 2)
    preprocess_column_short_sentences(df['claim'], 3)
    # Save file
    df.to_pickle(os.path.join(ROOT_DIR, '{0}_preprocessed.pkl'.format(file_name.split('.')[0])))


if __name__ == '__main__':
    file_name = sys.argv[1]
    # file_name = 'data/patent/part-000000000674.csv'
    # ROOT_DIR = os.path.dirname(os.path.abspath('/home/ubuntu/PycharmProjects/patent/requirements.txt'))
    main(ROOT_DIR, str(file_name))

