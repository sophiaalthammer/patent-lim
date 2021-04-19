# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import pandas as pd
import os
from definitions import ROOT_DIR
from ling_ana import preprocess_semscho


@pytest.fixture
def test_data():
    return pd.read_pickle(os.path.join(ROOT_DIR,'data/semanticscholar/s2-corpus-000_test.pkl'))


def test_clear_df(test_data):
    assert len(preprocess_semscho.clear_df(test_data)) == 11


def test_filter_df_en(test_data):
    df_cleared = preprocess_semscho.clear_df(test_data)
    assert len(preprocess_semscho.filter_df_en(df_cleared)) == 9
