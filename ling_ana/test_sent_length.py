import pytest
import pandas as pd
from ling_ana import sent_length_patent


@pytest.fixture
def test_data():
    return pd.read_csv(r'/home/ubuntu/PycharmProjects/patent/data/en_since_2000_a_unzip/test_data.csv')


@pytest.fixture
def sentences():
    return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


@pytest.fixture
def lengths():
    return list(range(5, 15))


def test_get_sentences(test_data):
    assert sent_length_patent.get_sentences(test_data['abstract'].iloc[4])[1] == \
           'The absorbent core may be substantially cellulose free or comprise a combination of particulate ' \
           'absorbent polymer material and wood pulp.'


def test_get_sentences2():
    assert sent_length_patent.get_sentences('Hi my name is\n What do you do.') == ['Hi my name is', ' What do you do.']


def test_get_sentences3():
    assert len(sent_length_patent.get_sentences('Hi I         want to have.')) == 2


def test_get_sentences4():
    assert len(sent_length_patent.get_sentences('Hi I        want to have.')) == 1


def test_count_sentence_length(test_data):
    assert sent_length_patent.count_sentence_length(sent_length_patent.get_sentences(test_data['abstract'].iloc[4])) == [25, 20, 15]


def test_get_sentence_length_column(test_data):
    assert sent_length_patent.get_sentence_lengths_column(test_data['claim'])[1][0] == 4


def test_remove_short_sentence_blocks(sentences, lengths):
    sent_short, len_short = sent_length_patent.remove_short_sentence_blocks(sentences, lengths)
    assert sent_short == ['5', '6', '7', '8', '9']


def test_remove_short_sentence_blocks2(test_data):
    sentences = sent_length_patent.get_sentences(test_data['description'].iloc[2])
    length = sent_length_patent.count_sentence_length(sentences)
    sent_short, leng_short = sent_length_patent.remove_short_sentence_blocks(sentences, length)
    assert leng_short == [5, 36, 28, 1, 4, 18, 4, 43, 26, 25, 23, 12, 24, 39, 25, 27, 33, 24, 28, 16, 31, 20, 17, 4, 20,
                          26, 38, 15, 27, 16, 16, 28, 29, 28, 5, 1, 15, 1, 19, 1, 15, 1, 14, 15, 40, 41, 40, 4, 17, 12,
                          33, 17, 54, 4, 43, 30, 4, 29, 33, 21, 2, 33, 10, 19, 8, 48, 17, 18, 31, 27, 13, 16, 12, 2, 14,
                          6, 34, 32, 23, 39, 35, 25, 28, 25, 19, 31, 19, 30, 22, 24, 29, 30, 13, 16, 38, 15, 22, 5, 37,
                          2, 8, 24, 2, 33, 1, 29, 2, 21, 32, 2, 17, 21, 4, 38, 22, 22, 2, 2, 17, 9, 25, 18, 25, 19, 23,
                          28, 34, 14, 9, 30, 8, 13, 21, 39, 21, 22, 22, 20, 19, 32, 8, 30, 15, 15, 15, 26, 34, 22, 50,
                          10, 27, 10, 13, 13, 15, 18, 24, 15, 17, 24, 15, 31, 19, 20, 26, 23, 17, 17, 18, 27, 26, 17, 1,
                          10, 1, 22, 17, 24, 35, 37, 29, 24, 30, 37, 15, 39, 11, 8, 31, 24, 22, 61, 12, 29, 19, 48, 38,
                          11, 1, 4, 12, 19, 19, 26, 4, 13, 15, 13, 1, 13, 27, 7, 26, 20, 19, 40, 22, 26, 13, 11, 15, 3,
                          13, 11, 10, 3, 11, 11, 10, 2, 10, 11, 22, 28, 35, 42, 16, 40, 10, 39, 13, 13, 17, 13, 12, 28,
                          16, 42, 10, 12, 10, 32, 56, 9, 44, 15, 12, 8, 2, 12, 1, 1, 17, 29, 16, 18, 31, 19, 17, 2, 2,
                          13, 3, 2, 12, 2, 10, 4, 2, 12, 13, 4, 10, 11, 3, 11, 15, 40, 28, 44, 22, 22, 34, 20, 19, 19,
                          10, 11, 4, 10, 2, 3, 10, 2, 2, 11, 3, 11, 12, 10, 2, 13, 12, 3, 10, 27, 12, 34, 23, 10, 17, 2,
                          5, 15, 2, 15, 13, 12, 4, 6, 10, 10, 10, 12, 5, 11, 6, 10, 10, 10, 43, 28, 8, 19, 43, 42, 42,
                          12, 64, 48, 15, 14]


def test_remove_short_sentences(sentences, lengths):
    sent_short, len_short = sent_length_patent.remove_short_sentences(sentences, lengths, 10)
    assert len_short == [10, 11, 12, 13, 14]


def test_remove_short_sentences(sentences, lengths):
    sent_short, len_short = sent_length_patent.remove_short_sentences(sentences, lengths, 10)
    assert sent_short == ['5', '6', '7', '8', '9']


def test_remove_long_sentences(sentences, lengths):
    sent_long, len_long = sent_length_patent.remove_long_sentences(sentences, lengths, 8)
    assert len_long == [5, 6, 7, 8]
