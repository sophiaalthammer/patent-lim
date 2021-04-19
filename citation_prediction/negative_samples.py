# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
import os
import sys
import time
import pickle
from definitions import ROOT_DIR


def main(ROOT_DIR: str, df_location: str, positive_pairs_loc: str):
    start = time.time()
    # Patent data
    df = pd.read_pickle(os.path.join(ROOT_DIR, df_location))

    print('start now with pairs')
    # positive Citation pairs
    cit = pd.read_csv(os.path.join(ROOT_DIR, positive_pairs_loc))

    index_pub, index_cit, cit_neg = get_citation_negative_examples(cit, df, positive_pairs_loc)
    end = time.time()
    print('Total time: {0}'.format(end-start))


def get_citation_negative_examples(cit: pd.DataFrame, df: pd.DataFrame, positive_pairs_loc: str):
    """
    Creates for the pub_numbers in the positive sample also negative samples
    Strategy: for all pub_numbers in the positive citations append a random publication name, remove the row if
    pub_number and negative_pub_number are the same or if positive and negative citation number are the same
    """
    cit['negative_cit_number'] = df['pub_number'].sample(len(cit), replace=True).values
    print(len(cit['pub_number']))
    index_drop_pub_neg_cit_same = cit[cit['pub_number'] == cit['negative_cit_number']].index
    if len(index_drop_pub_neg_cit_same) > 0:
        cit = cit.drop(index_drop_pub_neg_cit_same, axis=0)
        print(len(cit['pub_number']))
        print('Dropped because same pub number and now the length is: {0}'.format(len(cit['pub_number'])))
    index_drop_cit_neg_cit_same = cit[cit['cit_number'] == cit['negative_cit_number']].index
    if len(index_drop_pub_neg_cit_same) > 0:
        cit = cit.drop(index_drop_cit_neg_cit_same, axis=0)
        print('Dropped because same positive citation and now the length is: {0}'.format(len(cit['pub_number'])))
    print(len(cit['pub_number']))

    # drops the column of positive citations
    cit = cit.drop(['cit_number'], axis=1)
    # renames the column of negative citations in cit_number
    cit = cit.rename(columns={"negative_cit_number": "cit_number"})

    #cit = cit.sample(100000)
    # store citations with negatives
    with open(os.path.join(ROOT_DIR, "{0}/citations-only-type-x-with_claims_negatives.pkl".format(
            os.path.dirname(positive_pairs_loc))), "wb") as fp:
         pickle.dump(cit, fp)
    return index_drop_pub_neg_cit_same, index_drop_cit_neg_cit_same, cit


if __name__ == '__main__':
    df_location = sys.argv[1]
    positive_pairs_loc = sys.argv[2]
    #df_location = "data/citations/patent-contents-for-citations-wclaims/patent-contents-for-citations_en_claim_wclaims_all.pkl"
    # positive_pairs_loc = 'data/citations/patent-contents-for-citations-wclaims/citations-only-type-x-with_claims.csv'
    main(ROOT_DIR, str(df_location), str(positive_pairs_loc))
