# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
import numpy as np
import os
import sys
import gc
import pickle


def preprocess_csv_to_tsv(file_location: str, file, new_file_location: str, train_test: str):
    file_name = 'part-000000000{}.csv.gz'.format(file)
    df = pd.read_csv(os.path.join(file_location, file_name), compression='gzip', error_bad_lines=False)

    # Drop columns which we dont need
    df = df.drop(['publication_date', 'filing_date', 'priority_date', 'title_truncated', 'abstract_truncated',
                  'claim_truncated', 'description_truncated'], axis=1)

    # Remove rows with empty or NaN cells
    df.replace("", float("NaN"), inplace=True)
    df.dropna(subset=['title', 'abstract', 'claim', 'description', 'ipc', 'cpc'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Remove multiple whitespaces and tabs and newlines from text in columns of parts
    parts = ['title', 'abstract', 'claim', 'description']
    for part in parts:
        df[part] = df[part].str.replace('[ \t\n]+', ' ', regex=True)

    # Create new columns with top1 of IPC and CPC tag
    df['ipc_top1'] = get_subclass(df['ipc'])
    df['cpc_top1'] = get_subclass(df['cpc'])

    assert df.loc[df['ipc_top1'].isnull()].empty
    assert df.loc[df['cpc_top1'].isnull()].empty

    # This is how I could join them later for the classification when reading them in in the DataProcessor
    # print(df['title'].iloc[0] + '. ' + df['abstract'].iloc[0])

    # Store the data in the tsv file which will be used for training plus the unique tags contained in the file
    with open(os.path.join(new_file_location, 'ipc_unique_tags_0{0}.pkl'.format(file)), "wb") as fp:
        pickle.dump(list(set(df['ipc_top1'])), fp)
    with open(os.path.join(new_file_location, 'cpc_unique_tags_0{0}.pkl'.format(file)), "wb") as fp:
        pickle.dump(list(set(df['cpc_top1'])), fp)

    df.to_csv(os.path.join(new_file_location, '{0}_0{1}.tsv'.format(train_test, file)), sep='\t', index=False,
              header=False)

    del [df]
    gc.collect()


def create_cpc_lists(files, new_file_location: str, train_test: str):
    for file in files:
        df = pd.read_csv(os.path.join(new_file_location, '{0}_0{1}.tsv'.format(train_test, file)), sep='\t', header=None)

        # This is how I could join them later for the classification when reading them in in the DataProcessor
        # print(df['title'].iloc[0] + '. ' + df['abstract'].iloc[0])

        # Store the data in the tsv file which will be used for training plus the unique tags contained in the file
        with open(os.path.join(new_file_location, 'cpc_unique_tags_0{0}.pkl'.format(file)), "wb") as fp:
            pickle.dump(list(set(df[8])), fp)

        del [df]
        gc.collect()


def get_subclass(tag_column: pd.Series):
    """
    returns an numpy array containing the main subclass per patent (the most frequent class of all ipc tags) of
    the ipc or cpc tag depending on the input column for all patents in the dataframe
    :param tag_column: pd.Series in the format of a string like 'F15B13/4022;...'
    :return: numpy array with the main subclass for every row in dataframe
    """
    list_tags = tag_column.str.split(';')
    for i in range(len(list_tags)):
        tags = []
        if type(list_tags[i]) == list:
            for j in range(len(list_tags[i])):
                tags.append(list_tags[i][j][:4])
            list_tags[i] = max(set(tags), key=tags.count)
    return np.array(list_tags)


def merge_unique_tags_all_documents(file_location: str, files: list, ipc_cpc: str):
    """
    Merges all the lists of unique tags and saves the merged version as one pickle file with all unique tags
    """
    unique_tags = []
    for file in files:
        with open(os.path.join(file_location, '{0}_unique_tags_0{1}.pkl'.format(ipc_cpc, file)), "rb") as fp:
            unique_tag = pickle.load(fp)
            unique_tags.extend(unique_tag)
    unique_all_doc = list(set(unique_tags))
    with open(os.path.join(file_location, '{0}_all_unique_tags_files{1}-{2}.pkl'.format(ipc_cpc, files[0], files[-1])), "wb") as fp:
        pickle.dump(unique_all_doc, fp, protocol=0)
    return unique_all_doc


def main(file_location: str, new_file_location: str, train_test: str, start_files: int, end_files: int):
    files = ["{0:03}".format(i) for i in range(start_files, end_files)]
    for file in files:
        preprocess_csv_to_tsv(file_location, file, new_file_location, train_test)
    merge_unique_tags_all_documents(new_file_location, files, 'ipc')
    merge_unique_tags_all_documents(new_file_location, files, 'cpc')


if __name__ == '__main__':
    file_location = sys.argv[1]
    new_file_location = sys.argv[2]
    train_test = sys.argv[3]
    start_files = sys.argv[4]
    end_files = sys.argv[5]
    #file_location = '/home/ubuntu/Documents/thesis/data/patent_contents_en_since_2000_application_kind_a'
    #new_file_location = '/home/ubuntu/PycharmProjects/patent/emnlp_baseline3/data/ipc'
    #train_test = 'train'
    #start_files = 0
    #end_files = 3
    main(str(file_location), str(new_file_location), str(train_test), int(start_files), int(end_files))


