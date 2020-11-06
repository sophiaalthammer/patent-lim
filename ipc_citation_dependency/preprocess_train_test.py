import pandas as pd
import numpy as np
import os
import pickle
import time
from definitions import ROOT_DIR


def main():
    # Patent data
    start = time.time()
    df_location = "data/citations/patent-contents-for-citations-only-type_all_dict.pkl"
    with open(os.path.join(ROOT_DIR, df_location), "rb") as fp:
        dict = pickle.load(fp)
    df = pd.DataFrame(dict)

    # All unique tags contained in all patents
    with open(os.path.join(ROOT_DIR, 'data/citations/unique_tags_all_documents.pkl'), "rb") as fp:
         unique_tags_all_doc = pickle.load(fp)

    # Citation pairs
    #cit = pd.read_csv(os.path.join(ROOT_DIR, "data/citations/citations-only-type-x-v3.csv"))
    #cit = pd.read_pickle(os.path.join(ROOT_DIR, 'data/citations/citations_augmented_all.pkl'))
    #cit = cit.sample(100000)
    #print('Im Here')
    #cit.to_pickle(os.path.join(ROOT_DIR, "data/citations/citations-only-type-x-v3-sample100000_transitive_augmented.pkl"))
    #cit = pd.read_pickle(os.path.join(ROOT_DIR, "data/citations/citations-only-type-x-v3-sample100000.pkl"))

    # Positive Citation Vectors, index where there is a 1 in the vector of length 2*len(unique_tags_all_doc)
    #cit_vec = vectors_for_citation_pairs_fast(cit, df, unique_tags_all_doc)

    # Negative citation pairs
    cit_neg = pd.read_pickle(os.path.join(ROOT_DIR, "data/citations/citations-only-type-x-v3-sample100000_negative_trans_augmented.pkl"))
    cit_vec_neg = vectors_for_citation_pairs_fast(cit_neg, df, unique_tags_all_doc)

    end = time.time()
    print('Total time:{0}'.format(end - start))


def vectors_for_citation_pairs(cit: pd.DataFrame, df: pd.DataFrame, unique_tags_all_doc: list):
    """
    Returns for all citations paris in the dataframe cit the vector representation contained in df
    """
    vectors = []
    for i in range(len(cit)):
        pair = cit.iloc[i]
        pub_number_vector = df[df['pub_number'] == pair['pub_number']]['mul_hot'].reset_index()
        if len(pub_number_vector) >= 1:
            cit_number_vector = df[df['pub_number'] == pair['cit_number']]['mul_hot'].reset_index()
            if len(cit_number_vector) >= 1:
                cit_number_vector = [index + len(unique_tags_all_doc) for index in cit_number_vector['mul_hot'].iloc[0]]
                concat_vector = pub_number_vector['mul_hot'].iloc[0] + cit_number_vector
                vectors.append(concat_vector)
    with open(os.path.join(ROOT_DIR, "data/citations/cit_vec_sample10000_negative.pkl"), "wb") as fp:
         pickle.dump(vectors, fp)
    return vectors


def vectors_for_citation_pairs_fast(cit: pd.DataFrame, df: pd.DataFrame, unique_tags_all_doc: list):
    citing_patents = list(pd.merge(cit, df, on='pub_number')['mul_hot'])
    cit2 = cit.rename(columns={"pub_number": "pub_number2", "cit_number": "pub_number"})
    cited_patents = list(pd.merge(cit2, df, on='pub_number')['mul_hot'])
    cited_patents2 = [[i + len(unique_tags_all_doc) for i in list] for list in cited_patents]
    assert len(citing_patents) == len(cited_patents2)
    stacked = np.stack((citing_patents, cited_patents2), axis = 1)
    df_stacked = pd.DataFrame(stacked)
    vectors = list(df_stacked[0] + df_stacked[1])
    with open(os.path.join(ROOT_DIR, "data/citations/cit_vec_sample100000_negative_trans_augmented.pkl"), "wb") as fp:
         pickle.dump(vectors, fp)
    return vectors


if __name__ == '__main__':
    main()












