import pytest
import pandas as pd
import spacy
from ling_ana import hyphen_exp
import os
from definitions import ROOT_DIR


@pytest.fixture
def test_data_preprocessed():
    return pd.read_pickle(os.path.join(ROOT_DIR,'data/en_since_2000_a_unzip/test_data_preprocessed.pkl'))


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")


def test_get_hyphen_exp(nlp):
    assert hyphen_exp.get_hyphen_exp('a disk - shaped suspension-type insulator', nlp) == ['disk-shaped',
                                                                                           'suspension-type']


def test_get_hyphen_exp_column(test_data_preprocessed, nlp):
    conj = hyphen_exp.get_hyphen_exp_column(test_data_preprocessed['claim'], nlp)
    assert conj[0] == 'ice-resisting'


def test_get_share_conjunction_words(test_data_preprocessed, nlp):
    assert hyphen_exp.get_share_hyphen_exp(test_data_preprocessed['claim'], nlp) == 0.0058997050147492625
