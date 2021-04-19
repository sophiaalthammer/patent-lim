# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import spacy
import os
import sys
from ling_ana import sent_length_patent
from ling_ana.ipc_cpc_class_occurences import plot_hist
from ling_ana.count_words_patent import count_words
from ling_ana import noun_chunks_patent
from nltk.tokenize import RegexpTokenizer
from definitions import ROOT_DIR
import regex as re
import numpy as np


def remove_captions(location_input_file: str, location_output_file: str):
    """
    Opens a text file and removes lines which start with ' ='
    :param location_input_file: string containing the location and name of input file
    :param location_output_file: string containing the location and name of output file
    :return: None
    """
    with open(location_input_file, 'r') as input:
        with open(location_output_file, 'w') as output:
            for line in input:
                if not line.startswith(' ='):
                    output.write(line)


def get_hyphen_exp_wiki(sentences: list):
    """
    Gets all hyphen expressions contained in the sentences
    :param sentences: list containing strings
    :return: list containing hyphen expressions as string
    """
    hyphen_expressions = []
    for sent in sentences:
        tokens = sent.split()
        for i in range(len(tokens)):
            if tokens[i] == '@-@':
                hyphen_expressions.append(tokens[i - 1] + '-' + tokens[i + 1])
    return hyphen_expressions


def count_words_wiki(text: str):
    """
    Counts words contained in
    :param text:
    :return:
    """
    tokenizer = RegexpTokenizer(r'\w+')
    return len(tokenizer.tokenize(text))


def replace_hyphen_wiki(sentences: list):
    replaced = []
    for sent in sentences:
        replaced.append(re.sub('@-@', '-', sent))
    return replaced


def main(ROOT_DIR: str, location_input: str, location_output: str):
    """

    :param ROOT_DIR:
    :param location_input:
    :param location_output:
    :return:
    """
    # Preprocessing for sentence length analysis
    # Remove captions/subcaptions of text (captions/subcaptions start with ' =')
    location_input_file = os.path.join(ROOT_DIR, location_input)
    location_output_file = os.path.join(ROOT_DIR, location_output)

    remove_captions(location_input_file, location_output_file)

    # Open text file without captions/subcaptions
    text = open(location_output_file).read()

    # Get sentences of text file contained in a list
    sentences = sent_length_patent.get_sentences(text)
    sent_lengths = sent_length_patent.count_sentence_length(sentences)
    plot_hist(sent_lengths, 'Sentence Lengths Wikipedia', 'Sentence length')

    # Open text file with captions/subcaptions for noun chunk analysis
    text_with_captions = open(location_input_file).read()
    sentences_with_captions = sent_length_patent.get_sentences(text_with_captions)

    # Word count
    print('Word count Wikipedia text: {0}'.format(count_words(text_with_captions)))

    # Evaluate noun chunks
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 1500000
    sentences_w_captions_hyphen_replaced = replace_hyphen_wiki(sentences_with_captions)
    noun_chunks, noun_chunk_lengths = noun_chunks_patent.get_noun_chunk_column(sentences_w_captions_hyphen_replaced,
                                                                               nlp)
    plot_hist(np.array(noun_chunk_lengths), 'Noun Chunks Wikipedia', 'Noun chunks')
    print('Share of noun chunks in Wikipedia: {0}'.format(
        noun_chunks_patent.get_share_noun_chunks_column(sentences_w_captions_hyphen_replaced, nlp)))

    # Evaluate hyphen expressions
    hyphen_exp = get_hyphen_exp_wiki(sentences_with_captions)
    number_words = count_words_wiki(text_with_captions)
    # Multiplicated with 2 as count_words_wiki('guest @-@ starring') = 2
    share_hyphen_exp = len(hyphen_exp) * 2 / number_words
    print('Share Hyphen expressions in text: {0}'.format(share_hyphen_exp))


if __name__ == '__main__':
    location_input = sys.argv[1]
    location_output = sys.argv[2]
    # location_input = 'data/wikitext/wikitext-2-raw/wiki.test.raw'
    # location_output = 'data/wikitext/wikitext-2-raw/wiki.train.wo.captions.txt'
    # ROOT_DIR = os.path.dirname(os.path.abspath('/home/ubuntu/PycharmProjects/patent/requirements.txt'))
    main(ROOT_DIR, str(location_input), str(location_output))



