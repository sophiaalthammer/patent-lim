# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
import numpy as np
import string
import os
import sys
from ling_ana.ipc_cpc_class_occurences import plot_hist
import re
from ling_ana.sent_length_patent import plot_hist_2_distributions
from definitions import ROOT_DIR


def count_words(text: string):
    """
    Counts the words in a text which are separated by a whitespace (or by a '-'? To remove the discrepancy between
    number of words counted here and number of words counted in noun_chunks by spacy)
    :param text: string
    :return: int number of words in text
    """
    return len(re.split('-| ', remove_punctuation(text)))


def remove_punctuation(text: string):
    """
    removes the punctuation in a text, punctuation '-' wants to be kept to count the words like spacy (aber spacy counts
     '-' als einzelnes wort....)
    :param text: string
    :return: string without punctuation
    """
    return text.translate(str.maketrans('', '', string.punctuation.replace('-', '')))


def get_word_count_column(column: pd.Series):
    """
    counts the words for all cells in column of a dataframe
    :param column: pd.Series
    :return: list of word counts per cell
    """
    word_count = []
    for i in range(len(column)):
        word_count.append(count_words(column[i]))
    return word_count


def main(ROOT_DIR, file_name):
    """
    counts the number of the words in the different patent texts and compares them, removes outliers for claims
    (>8000 words) and descriptions (>10000 words)
    :param ROOT_DIR: root directory of data folder
    :param file_name: file name of the pickle file
    :return: None
    """
    df = pd.read_pickle(os.path.join(ROOT_DIR, file_name))
    df = df.head(2000)
    count = get_word_count_column(df['text'])
    print('Word count patent full text: {0}'.format(sum(count)))

    # Plot word count for column
    words_title = get_word_count_column(df['title'])
    print('Mean title length: {0}'.format(np.mean(words_title)))
    plot_hist(np.array(words_title), "Word counts", 'Title')

    words_abstract = get_word_count_column(df['abstract'])
    print('Mean abstract length: {0}'.format(np.mean(words_abstract)))
    plot_hist(np.array(words_abstract), "Word count", 'Abstract')
    plot_hist_2_distributions(np.array(words_abstract), "Abstract", np.array(words_title), "Title", "Word Count")

    # Outlier detection
    print('Outlier detection abstract maximal length: {0}'.format(max(words_abstract)))
    print('Longest abstract: {0}'.format(df['abstract'].iloc[words_abstract.index(max(words_abstract))]))

    # Plot word count column
    words_claim = get_word_count_column(df['claim'])
    words_descr = get_word_count_column(df['description'])
    plot_hist_2_distributions(np.array(words_claim), "Claims", np.array(words_descr), "Description", "Word Count")

    # Outlier detection
    print('Outlier detection claim maximal length: {0}'.format(max(words_claim)))
    print('Longest claim: {0}'.format(df['claim'].iloc[words_claim.index(max(words_claim))]))

    print('Outlier detection description maximal length: {0}'.format(max(words_descr)))
    print('Longest description: {0}'.format(df['description'].iloc[words_descr.index(max(words_descr))]))

    # Plot word counts, with and without outlier
    plot_hist(np.array(words_claim), "Word counts", 'Claims')
    plot_hist(np.array(words_descr), "Word counts", 'Description')

    # remove claims with more than 8000 words
    words_claim_removed = [i for i in words_claim if i < 8000]
    print(np.mean(words_claim_removed))
    plot_hist(np.array(words_claim_removed), "Word counts", 'Claims')

    # remove descriptions with more than 100000 words
    words_descr_removed = [i for i in words_descr if i < 100000]
    print(np.mean(words_descr_removed))
    plot_hist(np.array(words_descr_removed), "Word counts", 'Description')


if __name__ == '__main__':
    file_name = sys.argv[1]
    # file_name = 'data/patent/part-000000000674_preprocessed.pkl'
    # ROOT_DIR = os.path.dirname(os.path.abspath('/home/ubuntu/PycharmProjects/patent/requirements.txt'))
    main(ROOT_DIR, str(file_name))



