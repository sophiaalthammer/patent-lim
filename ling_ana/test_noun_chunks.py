# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import pandas as pd
from ling_ana import noun_chunks_patent
import spacy
import os
from definitions import ROOT_DIR


@pytest.fixture
def test_data():
    return pd.read_csv(os.path.join(ROOT_DIR,'data/en_since_2000_a_unzip/test_data.csv'))


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")


def test_get_noun_chunks(test_data, nlp):
    assert noun_chunks_patent.get_noun_chunks(test_data['abstract'].iloc[0], nlp)[2].text == 'A part'


def test_get_span_length(test_data, nlp):
    spans = noun_chunks_patent.get_noun_chunks(test_data['abstract'].iloc[0], nlp)
    assert noun_chunks_patent.get_span_length(spans)[2] == 2


def test_get_noun_chunk_column(test_data, nlp):
    chunks, lengths = noun_chunks_patent.get_noun_chunk_column(test_data['title'], nlp)
    assert lengths[10] == 2


def test_get_noun_chunk_column2(nlp):
    chunks, lengths = noun_chunks_patent.get_noun_chunk_column(
        [' Robert Boulter is an English film , television and theatre actor .',
         'He had a guest - starring role on the television series .']
        , nlp)
    assert chunks[6].text == 'a guest - starring role'


def test_get_share_long_noun_chunks(test_data, nlp):
    chunks, lengths = noun_chunks_patent.get_noun_chunk_column(test_data['claim'], nlp)
    assert noun_chunks_patent.get_share_long_spans(lengths) == 0.13960546282245828


def test_share_chunks_column(test_data, nlp):
    assert noun_chunks_patent.get_share_noun_chunks_column(test_data['abstract'], nlp) == 0.6555555555555556


def test_get_share_chunks_column(nlp):
    df = pd.DataFrame({'text': ['I like decent fish and a nice portion of potatoes.']})
    assert noun_chunks_patent.get_share_noun_chunks_column(df['text'], nlp) == 0.7

