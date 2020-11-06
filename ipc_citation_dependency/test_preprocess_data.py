import pytest
import pandas as pd
import os
from definitions import ROOT_DIR
from ling_ana.ipc_cpc_class_occurences import get_top_n_subclasses
from citation_prediction.preprocess_data import get_multiple_hot_vectors, get_unique_tags


@pytest.fixture
def test_data():
    return pd.read_pickle(os.path.join(ROOT_DIR, 'data/citations/test_data.pkl'))


@pytest.fixture
def unique_tags():
    return ['A61K', 'A61P', 'B05B', 'B25J', 'B65D', 'C07D', 'F01C']


def test_unique_tags(test_data):
    top_n_subclasses = get_top_n_subclasses(test_data['ipc'], 300)
    assert len(get_unique_tags(top_n_subclasses)) == 7


def test_get_multiple_hot_vectors(test_data, unique_tags):
    test_data['tags'] = get_top_n_subclasses(test_data['ipc'], 300)
    assert get_multiple_hot_vectors(test_data, unique_tags)[0] == [5, 0, 1]