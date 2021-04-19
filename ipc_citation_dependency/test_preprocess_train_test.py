# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pytest
from ipc_citation_dependency.preprocess_train_test import vectors_for_citation_pairs, vectors_for_citation_pairs_fast
import pandas as pd
import os
import pickle
from definitions import ROOT_DIR


@pytest.fixture
def cit_data():
    return pd.read_pickle(os.path.join(ROOT_DIR, 'data/citations/citations_test_data.pkl'))


@pytest.fixture
def unique_tags():
    with open(os.path.join(ROOT_DIR, 'data/citations/unique_tags_all_documents.pkl'), "rb") as fp:
         unique_tags_all_doc = pickle.load(fp)
    return unique_tags_all_doc


@pytest.fixture
def patent_data():
    df_location = 'data/citations/patent-contents-for-citations-only-type_all_dict.pkl'
    with open(os.path.join(ROOT_DIR, df_location), "rb") as fp:
        dict = pickle.load(fp)
    return pd.DataFrame(dict)

@pytest.fixture
def cit_vectors():
    with open(os.path.join(ROOT_DIR, "data/citations/test_cit_vec.pkl"), "rb") as fp:
         vectors = pickle.load(fp)
    return vectors


def test_vectors_for_citation(patent_data, cit_data, unique_tags):
    assert vectors_for_citation_pairs(cit_data, patent_data, unique_tags)[0] == [190, 859]


def test_vectors_for_citation_pairs_fast(unique_tags):
    df1 = pd.DataFrame({'pub_number': ['1', '2', '3', '4'], 'mul_hot': [[22, 23], [0, 1, 2], [1, 2, 3, 4], [3]]})
    cit2 = pd.DataFrame({'pub_number': ['1', '1', '1', '2'], 'cit_number': ['2', '3', '4', '3']})
    assert vectors_for_citation_pairs_fast(cit2, df1, unique_tags)[0] == [22, 23, 669, 670, 671]




