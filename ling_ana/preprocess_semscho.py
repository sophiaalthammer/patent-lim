# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
import json
import langdetect
import sys
import os
from definitions import ROOT_DIR


def read_json_file(input_file_location: str):
    """
    Reads in a file containing multiple json files with the keys 'id', 'title' and 'abstract' and returns a pd.Dataframe
    containing those 3 columns
    :param input_file_location: location of the input file
    :return:
    """
    id = []
    title = []
    abstract = []

    with open(os.path.join(ROOT_DIR,'data/semanticscholar/s2-corpus-000'), 'rb') as input_file:
        for line in input_file:
            data = json.loads(line)
            # if data['title'] != '' and data['paperAbstract'] != '':
            #    if langdetect.detect(data['title']) == 'en' and langdetect.detect(data['paperAbstract']) == 'en':
            id.append(data['id'])
            title.append(data['title'])
            abstract.append(data['paperAbstract'])

    return pd.DataFrame(list(zip(id, title, abstract)), columns=['id', 'title', 'abstract'])


def clear_df(df: pd.DataFrame):
    """
    filter out rows where title or abstract is ''
    :param df: pd.Dataframe containing the columns 'title' and 'abstract'
    :return: dataframe without empty rows for title and abstract
    """
    index_to_drop = df[df['title'] == ''].index
    df.drop(index_to_drop, inplace=True)

    index_to_drop2 = df[df['abstract'] == ''].index
    df.drop(index_to_drop2, inplace=True)
    return df.reset_index(drop=True)


def filter_df_en(df: pd.DataFrame):
    """
    Filter out rows where title of abstract is not english
    :param df: pd.Dataframe containing the columns 'title' and 'abstract'
    :return: dataframe without non-english rows for title and abstract
    """
    index_to_keep = []
    for i in range(len(df['id'])):
        try:
            if langdetect.detect(df['title'].iloc[i]) == 'en' and langdetect.detect(df['abstract'].iloc[i]) == 'en':
                index_to_keep.append(i)
        except Exception:
            pass
    df2 = df[df.index.isin(index_to_keep)]
    return df2.reset_index(drop=True)


def main(ROOT_DIR, file_name):
    """
    Preprocess Semantic Scholar Open Research Corpus
    # http://s2-public-api-prod.us-west-2.elasticbeanstalk.com/corpus/?_sm_au_=iVV4rbvSVkKRD7FqTRKNjKHWR8RV1
    # reqires column title and abstract
    :param ROOT_DIR: root directory of data folder
    :param file_name: file name of the Semantic Scholar file
    :return: None
    """
    # Load data
    df = read_json_file(os.path.join(ROOT_DIR, file_name))
    # filter out rows where title or abstract is ''
    df_cleared = clear_df(df)
    # Filter out rows where title of abstract is not english
    df_filtered = filter_df_en(df_cleared)
    # Save data
    df_filtered.to_pickle(os.path.join(ROOT_DIR, '{0}_clean.pkl'.format(file_name)))


if __name__ == '__main__':
    file_name = sys.argv[1]
    # file_name = 'data/semanticscholar/s2-corpus-000'
    # ROOT_DIR = os.path.dirname(os.path.abspath('/home/ubuntu/PycharmProjects/patent/requirements.txt'))
    main(ROOT_DIR, str(file_name))
