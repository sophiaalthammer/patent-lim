import pytest
import pandas as pd
from ling_ana import compare_vocab_encodings
from transformers import BertTokenizer
import os
from definitions import ROOT_DIR


@pytest.fixture
def test_data():
    return pd.read_pickle(os.path.join(ROOT_DIR,'data/en_since_2000_a_unzip/test_data_preprocessed.pkl'))


@pytest.fixture
def bert_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-cased')


@pytest.fixture
def scibert_tokenizer():
    return BertTokenizer.from_pretrained(os.path.join(ROOT_DIR,'models/scibert_scivocab_uncased'))


@pytest.fixture
def model_prefix():
    return os.path.join(ROOT_DIR,'models/sentencepiece/patents_part-000000000665_sp_preprocessed/patent_wdescr_5m_sent_30k_vocab')


def test_length_sp_model(test_data, model_prefix):
    lengths = compare_vocab_encodings.length_sp_model(test_data['abstract'], model_prefix)
    assert lengths[4] == 63


def test_length_bert(test_data, bert_tokenizer):
    lengths = compare_vocab_encodings.length_bert(test_data['abstract'], bert_tokenizer)
    assert lengths[0] == 180


def test_length_bert2(test_data, scibert_tokenizer):
    lengths = compare_vocab_encodings.length_bert(test_data['abstract'], scibert_tokenizer)
    assert lengths[0] == 164


def test_lengths_4_encodings(test_data, model_prefix, bert_tokenizer, scibert_tokenizer):
    df = compare_vocab_encodings.lengths_4_encodings(test_data['abstract'], model_prefix, bert_tokenizer, scibert_tokenizer)
    assert df['Bert Vocab'][0] == 13


def test_split_ratio_encoding_model(test_data, model_prefix, bert_tokenizer):
    models = [model_prefix, bert_tokenizer]
    assert compare_vocab_encodings.split_ratio_encoding_model(test_data['abstract'], models)[1] == 1.2280373831775702


def test_split_ratio_encoding_multiple_columns(test_data, model_prefix, bert_tokenizer):
    models = [model_prefix, bert_tokenizer]
    column_names = ['abstract', 'claim']
    model_names = ['Patent Vocab', 'Bert Vocab']
    assert compare_vocab_encodings.split_ratio_encoding_multiple_columns(
        test_data, column_names, models, model_names)['Patent Vocab'][0] == 1.108411214953271


