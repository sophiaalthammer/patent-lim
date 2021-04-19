# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import os
import pandas as pd
from format_text_pretrain.format_pretrain_data import preprocess_text_column, split_sentences_nltk, split_sentences
from definitions import ROOT_DIR


@pytest.fixture
def test_data():
    return pd.DataFrame({'title': ['first patent', 'second patent'], 'abstract':['Here   goes an abstract. \n \n I like  that. ',
                                                                                 'Here goes the second abstract'],
                         'claim': ['Here   goes a claim. \n \n I like  that. ', 'Here goes the second claim'],
                         'description': ['Here   goes a description. \n \n I like  that. ', 'Here goes      the second description']})


def preprocessed_data():
    return pd.read_pickle('/home/ubuntu/Documents/thesis/data/patent_contents_en_since_2000_application_kind_a/test_df.pkl')


#def test_preprocess_text_column(test_data):
#    preprocessed = preprocess_text_column(test_data)
#    assert preprocessed['text'].iloc[0] == 'first patent\nHere goes an abstract. \n I like that. \nHere goes a claim.' \
#                                           ' \n I like that. \nHere goes a description. \n I like that.'
