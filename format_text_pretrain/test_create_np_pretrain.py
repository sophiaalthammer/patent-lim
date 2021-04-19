# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import os
import spacy
import numpy as np
from definitions import ROOT_DIR
from format_text_pretrain.create_np_pretrain import get_mapping, get_noun_phrase_vector, main


@pytest.fixture
def test_string():
    return 'I like big and clumsy frogs, but I slightly hate And you?'


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def test_input_file():
    return 'bert/data/npvector_test_text.txt'


@pytest.fixture
def test_output_file():
    return 'bert/data/npvector_test_vectors.txt'


@pytest.fixture
def mapping():
    return get_mapping()


def test_get_noun_phrase_vector(nlp, mapping, test_string):
    assert get_noun_phrase_vector(nlp, test_string, mapping) == [1, 0, 7, 13, 17, 1, 0, 0, 1, 0, 0,13, 1, 0]


def test_get_noun_phrase_vector_line(nlp, mapping):
    with open(os.path.join(ROOT_DIR, 'bert/data/first1000part674.txt'), 'r') as f:
        line = f.readline()
        assert get_noun_phrase_vector(nlp, line.strip(), mapping) == [16, 16, 1, 0, 1]


def test_main(test_input_file, test_output_file):
    main(test_input_file, test_output_file)
    with open(os.path.join(ROOT_DIR, test_output_file), 'r') as f:
        lines = f.readlines()
        assert lines[0] == '[1, 0, 7, 13, 17, 1, 0, 0, 1, 0, 0, 13, 1, 0]\n'
        assert lines[1] == '\n'
