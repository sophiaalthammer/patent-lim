import pytest
from ipc_citation_dependency.train_classifier import get_x_data_from_citation_vectors
import os
import pickle
from definitions import ROOT_DIR


@pytest.fixture
def unique_tags():
    with open(os.path.join(ROOT_DIR, 'data/citations/unique_tags_all_documents.pkl'), "rb") as fp:
         unique_tags_all_doc = pickle.load(fp)
    return unique_tags_all_doc


@pytest.fixture
def cit_vectors():
    with open(os.path.join(ROOT_DIR, "data/citations/test_cit_vec.pkl"), "rb") as fp:
         vectors = pickle.load(fp)
    return vectors


def test_get_x_data_from_citation_vectors(cit_vectors, unique_tags):
    assert get_x_data_from_citation_vectors(cit_vectors, unique_tags).shape == (4, 1338)


