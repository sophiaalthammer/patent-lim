import pytest
import pandas as pd
from ling_ana import ipc_cpc_class_occurences
import numpy as np
import os
from definitions import ROOT_DIR


@pytest.fixture
def test_data():
    return pd.read_csv(os.path.join(ROOT_DIR,'data/en_since_2000_a_unzip/test_data.csv'))


def test_year(test_data):
    assert ipc_cpc_class_occurences.year(test_data['publication_date'][0]) == np.array(2014)


def test_year2(test_data):
    assert ipc_cpc_class_occurences.year(test_data['publication_date'])[1] == 2018


def test_get_subclass(test_data):
    assert ipc_cpc_class_occurences.get_subclass(test_data['ipc'])[0] == 'H01B'


def test_get_top_n_subclasses(test_data):
    assert ipc_cpc_class_occurences.get_top_n_subclasses(test_data['ipc'], 2)[3] == ['B60K', 'H02K']


def test_get_section(test_data):
    assert ipc_cpc_class_occurences.get_section(test_data['cpc'])[0] == 'H'


def test_get_class(test_data):
    assert ipc_cpc_class_occurences.get_class(test_data['ipc'])[0] == 'H01'


def test_get_all_subclasses(test_data):
    assert ipc_cpc_class_occurences.get_all_subclasses(test_data['cpc'])[1][1] == 'H04L'
