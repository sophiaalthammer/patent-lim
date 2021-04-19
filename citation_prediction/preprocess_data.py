# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
import os
import pickle
from definitions import ROOT_DIR
from ling_ana.ipc_cpc_class_occurences import get_top_n_subclasses


def main(ROOT_DIR: str):
    index = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    get_unique_tags_all_documents(index)
    merge_unique_tags_all_documents(index)
    with open(os.path.join(ROOT_DIR, 'data/citations/unique_tags_all_documents.pkl'), "rb") as fp:
        unique_all_doc = pickle.load(fp)

    save_multiple_hot_vectors_for_all_indices(index, unique_all_doc)
    concat_and_save_patent_data(index)


def save_multiple_hot_vectors_for_all_indices(index: list, unique_tags_all_doc: list):
    """
    Opens all patent files contained in the index list, drops the columns we dont need and adds the columns tags and
    mul_hot, which contain for each patent the indexes of the unique tags
    """
    for i in index:
        df_location = 'data/citations/patent-contents-for-citations-only-type-x-v3_part-0000000000{0}'.format(i)
        dict = df_add_mul_hot(df_location, unique_tags_all_doc)
        with open(os.path.join(ROOT_DIR, df_location + '_mul_hot.pkl'), "wb") as fp:
            pickle.dump(dict, fp)


def concat_and_save_patent_data(index: list):
    """
    Concatenates the different dictionaries to one, drops duplicate pub_numbers and saves the file as a dictionary
    """
    df = pd.DataFrame(columns=['pub_number', 'tags', 'mul_hot'])
    for i in index:
        df_location = 'data/citations/patent-contents-for-citations-only-type-x-v3_part-0000000000{0}_mul_hot.pkl'.format(i)
        with open(os.path.join(ROOT_DIR, df_location), "rb") as fp:
            d1 = pickle.load(fp)
            df1 = pd.DataFrame(d1)
            df = pd.concat([df, df1], ignore_index=True)

    df.drop_duplicates(subset="pub_number", keep='first', inplace=True)
    dict = df.to_dict('list')
    with open(os.path.join(ROOT_DIR, "data/citations/patent-contents-for-citations-only-type_all_dict.pkl"), "wb") as fp:
         pickle.dump(dict, fp)


def df_add_mul_hot(df_location: str, unique_tags):
    """
    Drops unneeded columns in dataframe, creates a column containing the tags of the subclasses and adds a column with
    the index of the multiple hot vectors using the list of unique tags
    """
    df = pd.read_csv(os.path.join(ROOT_DIR, '{0}.csv-1'.format(df_location)))
    df = df.drop(['pub_date', 'filing_date', 'priority_date', 'title', 'abstract'], axis=1)
    top_n_subclasses = get_top_n_subclasses(df['ipc'], 300)
    df['tags'] = top_n_subclasses
    mul_hot = get_multiple_hot_vectors(df, unique_tags)
    return {'pub_number': list(df['pub_number']), 'tags': top_n_subclasses, 'mul_hot': mul_hot}


def get_unique_tags(top_n_subclasses: list):
    """
    Gets the unique tags in all top n subclasses
    """
    all_tags = []
    for i in range(len(top_n_subclasses)):
        all_tags.extend(top_n_subclasses[i])
    return sorted(set(all_tags))


def get_multiple_hot_vectors(df: pd.DataFrame, unique_tags: list):
    """
    get the indices for the multiple hot vectors, index revers to the index which is a 1
    """
    multiple_hot = []
    for i in range(len(df['tags'])):
        indexes_of_tags = [unique_tags.index(tag) for tag in df['tags'][i]]
        mul_hot = [1 if i in indexes_of_tags else 0 for i in range(len(unique_tags))]
        multiple_hot.append(indexes_of_tags)
    return multiple_hot


def get_unique_tags_all_documents(index: list):
    """
    Gets the unique tags for all documents contained in the list index and saves them as a pickle file
    """
    for i in index:
        df_location = 'data/citations/patent-contents-for-citations-only-type-x-v3_part-0000000000{0}'.format(i)
        df = pd.read_csv(os.path.join(ROOT_DIR, '{0}.csv-1'.format(df_location)))
        df = df.drop(['pub_date', 'filing_date', 'priority_date', 'title', 'abstract'], axis=1)
        top_n_subclasses = get_top_n_subclasses(df['ipc'], 300)
        unique_tags = get_unique_tags(top_n_subclasses)
        with open(os.path.join(ROOT_DIR, 'data/citations/unique_tags_{0}.pkl'.format(i)), "wb") as fp:
            pickle.dump(unique_tags, fp)


def merge_unique_tags_all_documents(index: list):
    """
    Merges all the lists of unique tags and saves the merged version as one pickle file with all unique tags
    """
    unique_tags = []
    for i in index:
        location = 'data/citations/unique_tags_{0}.pkl'.format(i)
        with open(os.path.join(ROOT_DIR, location), "rb") as fp:
            unique_tag = pickle.load(fp)
            unique_tags.extend(unique_tag)
    unique_all_doc = list(set(unique_tags))
    with open(os.path.join(ROOT_DIR, 'data/citations/unique_tags_all_documents.pkl'), "wb") as fp:
        pickle.dump(unique_all_doc, fp)
    return unique_all_doc


if __name__ == '__main__':
    main()















