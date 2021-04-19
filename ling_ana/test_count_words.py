# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pytest
from ling_ana import count_words_patent
import pandas as pd
import os
from definitions import ROOT_DIR


@pytest.fixture
def test_data():
    return pd.read_csv(os.path.join(ROOT_DIR,'data/en_since_2000_a_unzip/test_data.csv'))


def test_count_words(test_data):
    assert count_words_patent.count_words(test_data['abstract'].iloc[4]) == 60


def test_count_words2():
    assert count_words_patent.count_words('a disk-shaped suspension-type insulator') == 6


def test_remove_punctuation():
    assert count_words_patent.remove_punctuation('Hi! I, me. you?') == 'Hi I me you'


def test_get_word_count_column(test_data):
    assert count_words_patent.get_word_count_column(test_data['title'])[0] == 6
