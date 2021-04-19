# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import spacy
import os
import random
import unicodedata
import create_pretraining_data_lim


@pytest.fixture
def test_line():
    return unicode('I like big and clumsy frogs fetmo, but I slightly hate And you fetmo?')


@pytest.fixture
def test_np_line():
    return [1, 0, 7, 13, 17, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]


@pytest.fixture
def test_tokens_bert():
    return [u'I', u'like', u'big', u'and', u'clumsy', u'frogs', u'f', u'##et', u'##mo', u',', u'but', u'I',
            u'slightly', u'hate', u'And', u'you', u'f', u'##et', u'##mo', u'?']


@pytest.fixture
def test_nps_bert():
    return [1, 0, 7, 13, 17, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]


@pytest.fixture
def test_tokens_bert_unknown():
    return [u'I', u'like', u'big', u'and', u'clumsy', u'[UNK]', u'f', u'##et', u'##mo', u',', u'but', u'I',
            u'slightly', u'hate', u'And', u'you', u'f', u'##et', u'##mo', u'?']


@pytest.fixture
def test_tokens_bert_merged():
    return [u'I', u'like', u'big', u'and', u'clumsy', u'frogs fetmo', u',', u'but', u'I',
            u'slightly', u'hate', u'And', u'you', u'f', u'##et', u'##mo', u'?']


@pytest.fixture
def test_tokens_split_end():
    return [u'I', u'f', u'##et', u'##mo']


@pytest.fixture
def test_nps_split_end():
    return [0, 1]


@pytest.fixture
def test_line_split_end():
    return unicode('I fetmo')


@pytest.fixture
def test_tokens_unknown_end():
    return [u'I', u'[UNK]']


@pytest.fixture
def test_tokenizer_spacy():
    nlp = spacy.load("en_core_web_sm")
    return nlp.Defaults.create_tokenizer(nlp)


@pytest.fixture
def test_rng():
    return random.Random('12345')


def test_extend_unkown_end(test_tokenizer_spacy, test_line_split_end, test_nps_split_end, test_tokens_unknown_end):
    assert create_pretraining_data_lim.extend_np(test_line_split_end, test_tokens_unknown_end, test_nps_split_end,
                                                 test_tokenizer_spacy) == [0, 1]


def test_extend_np_with_split(test_line, test_tokens_bert, test_tokenizer_spacy, test_np_line):
    assert create_pretraining_data_lim.extend_np(test_line, test_tokens_bert, test_np_line, test_tokenizer_spacy) == \
           [1, 0, 7, 13, 17, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]


def test_extend_np_empty_list(test_tokenizer_spacy):
    assert create_pretraining_data_lim.extend_np(u'', [], '', test_tokenizer_spacy) == []


def test_extend_np_merged(test_line, test_tokens_bert_merged, test_tokenizer_spacy, test_np_line):
    assert create_pretraining_data_lim.extend_np(test_line, test_tokens_bert_merged, test_np_line, test_tokenizer_spacy) == \
           [1, 0, 7, 13, 17, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    # this creates an assertion error which shows the importance of the assumption that a the tokenization of bert is
    # more granular than the spacy tokenization


#def test_extend_np_with_unk(test_line, test_tokens_bert_unknown, test_tokenizer_spacy, test_np_line):
#    assert create_pretraining_data_lim.extend_np(test_line, test_tokens_bert_unknown, test_np_line, test_tokenizer_spacy) == \
#           [1, 0, 7, 13, 17, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]


def test_end_split(test_tokens_split_end, test_nps_split_end, test_line_split_end, test_tokenizer_spacy):
    assert create_pretraining_data_lim.extend_np(test_line_split_end, test_tokens_split_end, test_nps_split_end,
                                                 test_tokenizer_spacy) == [0, 1, 1, 1]

